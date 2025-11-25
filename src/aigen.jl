using HTTP
using JSON3

"""
    aigen_raw(prompt, provider_model::String; 
              schema::Union{AbstractRequestSchema, Nothing} = nothing,
              api_key::Union{String, Nothing} = nothing,
              sys_msg = nothing,
              streamcallback::Union{Nothing, AbstractLLMStream} = nothing,
              kwargs...)

Generate text using a specific provider and model, returning raw API response and parsing components.

This function is useful for:
- Testing equivalence between streaming and non-streaming responses
- Debugging API response formats
- Custom response processing

# Returns
- `NamedTuple`: Contains `(result, schema, provider_info, model_id, provider_endpoint, elapsed)`

# Example
```julia
# Compare streaming vs non-streaming raw responses
raw_stream = aigen_raw("Hello", "anthropic:claude-3-sonnet"; streamcallback=HttpStreamCallback())
raw_normal = aigen_raw("Hello", "anthropic:claude-3-sonnet")
```
"""
function aigen_raw(prompt, provider_model::String; 
                   schema::Union{AbstractRequestSchema, Nothing} = nothing,
                   api_key::Union{String, Nothing} = nothing,
                   sys_msg = nothing,
                   streamcallback::Union{Nothing, AbstractLLMStream} = nothing,
                   kwargs...)
    
    # Resolve alias to full provider:model format
    resolved_model = resolve_model_alias(provider_model)
    
    provider_info, model_id, provider_endpoint = parse_provider_model(resolved_model)
    
    # Record start time for elapsed calculation
    start_time = time()
    
    result = _aigen_core(prompt, provider_info, model_id, provider_endpoint; schema, api_key, sys_msg, streamcallback, kwargs...)
    
    elapsed = time() - start_time
    
    # Get schema for extraction if not provided
    if schema === nothing
        schema = get_provider_schema(provider_info, model_id)
    end
    
    return (
        result = result,
        schema = schema,
        provider_info = provider_info,
        model_id = model_id,
        provider_endpoint = provider_endpoint,
        elapsed = elapsed
    )
end

"""
    aigen(prompt, provider_model::String; ...)
    aigen(prompt, config::ModelConfig; ...)

Generate text using a specific provider and model, or a ModelConfig.

# Arguments
- `prompt`: The input prompt (String or Vector of message dicts)
- `provider_model::String`: Format "Provider:model/slug" (e.g., "Together:moonshotai/kimi-k2-thinking")
- `config::ModelConfig`: Model configuration with slug and parameters

# Keyword Arguments
- `schema::Union{AbstractRequestSchema, Nothing}`: Request schema to use (auto-detected if not provided)
- `api_key::Union{String, Nothing}`: Provider-specific API key (auto-detected from env if not provided)
- `sys_msg`: System message/instruction
- `streamcallback::Union{Nothing, AbstractLLMStream}`: Stream callback for real-time processing
- `kwargs...`: Additional API parameters

# Returns
- `AIMessage`: Generated response with metadata (cost, tokens, etc.)

# Example
```julia
# Using string slug
response = aigen("Write a haiku about Julia programming", "Together:moonshotai/kimi-k2-thinking")

# Using ModelConfig
config = ModelConfig("openai:openai/gpt-5.1"; temperature=0.7, max_tokens=1000)
response = aigen("Hello", config)

# Using system message
response = aigen("Hello", "Anthropic:claude-3-sonnet"; sys_msg="You are a helpful assistant")

# Using streaming
using OpenRouter
callback = HttpStreamCallback(; out=stdout)
response = aigen("Count to 10", "anthropic:anthropic/claude-haiku-4.5"; streamcallback=callback)
```
"""
function aigen(prompt, provider_model::String; 
               schema::Union{AbstractRequestSchema, Nothing} = nothing,
               api_key::Union{String, Nothing} = nothing,
               sys_msg = nothing,
               streamcallback::Union{Nothing, AbstractLLMStream} = nothing,
               verbose=nothing,
               kwargs...)
    
    # Get raw response and all components
    raw = aigen_raw(prompt, provider_model; schema, api_key, sys_msg, streamcallback, verbose, kwargs...)
    
    # Extract content and build AIMessage
    content = extract_content(raw.schema, raw.result)
    finish_reason = extract_finish_reason(raw.schema, raw.result)
    tokens = extract_tokens(raw.schema, raw.result)
    cost = calculate_cost(raw.provider_endpoint, tokens)
    reasoning = extract_reasoning(raw.schema, raw.result)
    
    return AIMessage(
        content=content,
        finish_reason=finish_reason,
        tokens=tokens,
        elapsed=raw.elapsed,
        cost=cost,
        reasoning=reasoning
    )
end

# ModelConfig overload
function aigen(prompt, config::ModelConfig;
               api_key::Union{String, Nothing} = nothing,
               sys_msg = nothing,
               streamcallback::Union{Nothing, AbstractLLMStream} = nothing,
               verbose=nothing,
               kwargs...)
    
    # Extract config and merge kwargs
    slug, schema, merged_kwargs = extract_config(config, kwargs)
    
    # Use schema from config if not overridden
    schema = coalesce(get(kwargs, :schema, nothing), schema)
    
    return aigen(prompt, slug;
                 schema=schema,
                 api_key=api_key,
                 sys_msg=sys_msg,
                 streamcallback=streamcallback,
                 verbose=verbose,
                 merged_kwargs...)
end

"""
Core function that handles both streaming and non-streaming API calls.
"""
function _aigen_core(prompt, provider_info::ProviderInfo, model_id::AbstractString, provider_endpoint::ProviderEndpoint; 
                     schema::Union{AbstractRequestSchema, Nothing} = nothing,
                     api_key::Union{AbstractString, Nothing} = nothing,
                     sys_msg = nothing,
                     streamcallback::Union{Nothing, AbstractLLMStream} = nothing,
                     verbose = nothing,
                     kwargs...)
    
    # Get schema (prefer explicit, otherwise provider default with model awareness)
    protocolSchema = schema === nothing ? get_provider_schema(provider_info, model_id) : schema
    
    # Get API key (prefer explicit, otherwise env var)
    if api_key === nothing
        if provider_info.api_key_env_var === nothing
            api_key = ""  # providers like Ollama that don't require auth
        else
            api_key = get(ENV, provider_info.api_key_env_var, "")
            isempty(api_key) && throw(ArgumentError("API key not found in environment variable $(provider_info.api_key_env_var)"))
        end
    end
    
    # Build request payload using schema (pass stream as positional argument)
    stream_flag = streamcallback !== nothing
    payload = build_payload(protocolSchema, prompt, model_id, sys_msg, stream_flag; kwargs...)
    !isnothing(verbose) && !!verbose && @show payload
    
    # Build headers
    headers = build_headers(provider_info, api_key)
    
    # Build URL using schema
    url = build_url(protocolSchema, provider_info.base_url, model_id, stream_flag)
    
    # Branch based on streaming
    if streamcallback === nothing
        # Non-streaming request
        response = HTTP.post(url, headers, JSON3.write(payload))
        
        response.status != 200 && error("API request failed with status $(response.status): $(String(response.body))")
        
        return JSON3.read(response.body, Dict)
    else
        # Configure stream callback with schema and provider info
        configure_stream_callback!(streamcallback, protocolSchema, provider_info, provider_endpoint)
        
        # Streaming request
        response = streamed_request!(streamcallback, url, headers, JSON3.write(payload))
        
        response.status != 200 && error("API request failed with status $(response.status): $(String(response.body))")
        
        return JSON3.read(response.body, Dict)
    end
end

# Parse "Provider:author/model_id" into (ProviderInfo, "transformed_model_id", ProviderEndpoint)
function parse_provider_model(provider_model::AbstractString)
    parts = split(provider_model, ":", limit=2)
    length(parts) == 2 || throw(ArgumentError("Modelname must be in format \"provider:author/model_id\", got \"$provider_model\""))
    
    provider_name = parts[1]
    model_id = parts[2]
    
    # Get provider info
    provider_info = get_provider_info(lowercase(provider_name))
    provider_info === nothing && throw(ArgumentError("Unknown provider: $provider_name. Available: $(join(sort(list_known_providers()), ", "))"))
    
    # Get the cached model with endpoints
    cached_model = get_model(model_id; fetch_endpoints=true)
    if cached_model === nothing
        # Special handling for local providers (e.g. Ollama) that don't have OpenRouter metadata
        if lowercase(provider_name) == "ollama"
            transformed_model_id = transform_model_name(provider_info, model_id)
            stub_endpoint = create_stub_endpoint(provider_name, model_id)
            return provider_info, transformed_model_id, stub_endpoint
        end
        
        # For other providers, this is an error - show helpful message
        available_models = list_models(lowercase(provider_name))
        model_ids = [m.id for m in available_models]
        hint = "\nHint: Available models for $provider_name: $(join(model_ids, ", "))"
        throw(ArgumentError("Model not found: $model_id. Use update_db() to refresh the model database.$hint"))
    end
    
    provider_lower = lowercase(provider_name)
    # Find the specific endpoint for this provider
    provider_endpoint = nothing
    if cached_model.endpoints !== nothing
        for endpoint in cached_model.endpoints.endpoints
            if lowercase(endpoint.provider_name) == provider_lower || endpoint.tag == provider_lower 
                provider_endpoint = endpoint
                break
            end
        end
    end
    
    if provider_endpoint === nothing
        throw(ArgumentError("Provider $provider_name does not host model $model_id. Available providers: $(join([ep.provider_name for ep in cached_model.endpoints.endpoints], ", "))"))
    end
    
    # Transform model name according to provider rules (this now handles prefix removal internally)
    transformed_model_id = transform_model_name(provider_info, model_id)
    
    return provider_info, transformed_model_id, provider_endpoint
end