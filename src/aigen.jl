using HTTP
using JSON3

"""
    aigen(prompt, provider_model::String; 
          schema::Union{AbstractRequestSchema, Nothing} = nothing,
          api_key::Union{String, Nothing} = nothing,
          sys_msg = nothing,
          stream_callback::Union{Nothing, AbstractLLMStream} = nothing,
          kwargs...)

Generate text using a specific provider and model.

# Arguments
- `prompt`: The input prompt (String or Vector of message dicts)
- `provider_model::String`: Format "Provider:model/slug" (e.g., "Together:moonshotai/kimi-k2-thinking")

# Keyword Arguments
- `schema::Union{AbstractRequestSchema, Nothing}`: Request schema to use (auto-detected if not provided)
- `api_key::Union{String, Nothing}`: Provider-specific API key (auto-detected from env if not provided)
- `sys_msg`: System message/instruction
- `stream_callback::Union{Nothing, AbstractLLMStream}`: Stream callback for real-time processing
- `kwargs...`: Additional API parameters

# Returns
- `String`: Generated text response

# Example
```julia
response = aigen("Write a haiku about Julia programming", "Together:moonshotai/kimi-k2-thinking")
println(response)

# Using system message
response = aigen("Hello", "Anthropic:claude-3-sonnet"; sys_msg="You are a helpful assistant")

# Using streaming
using OpenRouter
callback = HttpStreamCallback(; out=stdout)
response = aigen("Count to 10", "anthropic:anthropic/claude-haiku-4.5"; stream_callback=callback)
```
"""
function aigen(prompt, provider_model::String; 
               schema::Union{AbstractRequestSchema, Nothing} = nothing,
               api_key::Union{String, Nothing} = nothing,
               sys_msg = nothing,
               stream_callback::Union{Nothing, AbstractLLMStream} = nothing,
               kwargs...)
    
    provider_info, model_id = parse_provider_model(provider_model)
    result = _aigen_core(prompt, provider_info, model_id; schema=schema, api_key=api_key, sys_msg=sys_msg, stream_callback=stream_callback, kwargs...)
    
    # Get schema for extraction if not provided
    if schema === nothing
        schema = get_provider_schema(provider_info)
    end
    
    # Extract content using schema
    return extract_content(schema, result)
end

"""
    aigen_raw(prompt, provider_model::String; 
              schema::Union{AbstractRequestSchema, Nothing} = nothing,
              sys_msg = nothing,
              stream_callback::Union{Nothing, AbstractLLMStream} = nothing,
              kwargs...)

Generate text and return the raw API response as a dictionary.
Useful for accessing usage statistics, multiple choices, etc.
"""
function aigen_raw(prompt, provider_model::String; 
                   schema::Union{AbstractRequestSchema, Nothing} = nothing,
                   api_key::Union{String, Nothing} = nothing,
                   sys_msg = nothing,
                   stream_callback::Union{Nothing, AbstractLLMStream} = nothing,
                   kwargs...)
    
    provider_info, model_id = parse_provider_model(provider_model)
    return _aigen_core(prompt, provider_info, model_id; schema=schema, api_key=api_key, sys_msg=sys_msg, stream_callback=stream_callback, kwargs...)
end

"""
Core function that handles both streaming and non-streaming API calls.
"""
function _aigen_core(prompt, provider_info::ProviderInfo, model_id::AbstractString; 
                     schema::Union{AbstractRequestSchema, Nothing} = nothing,
                     api_key::Union{AbstractString, Nothing} = nothing,
                     sys_msg = nothing,
                     stream_callback::Union{Nothing, AbstractLLMStream} = nothing,
                     kwargs...)
    
    # Get schema (prefer explicit, otherwise provider default)
    protocolSchema = schema === nothing ? get_provider_schema(provider_info) : schema
    
    # Get API key (prefer explicit, otherwise env var)
    api_key = api_key === nothing ? get(ENV, provider_info.api_key_env_var, "") : api_key
    isempty(api_key) && throw(ArgumentError("API key not found in environment variable $(provider_info.api_key_env_var)"))
    
    # Build request payload using schema (pass stream as positional argument)
    stream_flag = stream_callback !== nothing
    payload = build_payload(protocolSchema, prompt, model_id, sys_msg, stream_flag; kwargs...)
    
    # Build headers
    headers = build_headers(provider_info, api_key)
    
    # Build URL using schema
    url = build_url(protocolSchema, provider_info.base_url, model_id, stream_flag)
    
    # Branch based on streaming
    if stream_callback === nothing
        # Non-streaming request
        @show payload
        @show headers
        @show url
        response = HTTP.post(url, headers, JSON3.write(payload))
        
        response.status != 200 && error("API request failed with status $(response.status): $(String(response.body))")
        
        return JSON3.read(response.body, Dict)
    else
        # Configure stream callback schema if not set
        stream_callback.schema === nothing && (stream_callback.schema = protocolSchema)
        
        # Streaming request
        response = streamed_request!(stream_callback, url, headers, JSON3.write(payload))
        
        response.status != 200 && error("API request failed with status $(response.status): $(String(response.body))")
        
        return JSON3.read(response.body, Dict)
    end
end

# Parse "Provider:model/slug" into (ProviderInfo, "transformed_model_id")
function parse_provider_model(provider_model::AbstractString)
    parts = split(provider_model, ":", limit=2)
    length(parts) == 2 || throw(ArgumentError("provider_model must be in format \"Provider:model/slug\", got \"$provider_model\""))
    
    provider_name = parts[1]
    model_id = parts[2]
    
    # Get provider info
    provider_info = get_provider_info(lowercase(provider_name))
    provider_info === nothing && throw(ArgumentError("Unknown provider: $provider_name. Available: $(join(sort(list_known_providers()), ", "))"))
    
    # Strip redundant provider prefix from model_id if present
    provider_lower = lowercase(provider_name)
    if startswith(model_id, provider_lower * "/")
        model_id = model_id[(length(provider_lower) + 2):end]  # +2 for the "/"
    end
    
    # Transform model name according to provider rules
    transformed_model_id = transform_model_name(provider_info, model_id)
    
    return provider_info, transformed_model_id
end