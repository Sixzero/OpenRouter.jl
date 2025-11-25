
"""
Provider information for OpenRouter API providers.
Contains general information about each provider, not request-specific configuration.
"""

struct ProviderInfo
    base_url::String
    auth_header_format::String        # "Bearer", "Api-Key", "x-api-key", etc.
    api_key_env_var::Union{String,Nothing}  # Full env var name, e.g. "OPENAI_API_KEY"
    default_headers::Dict{String,String}    # Headers always sent with requests
    model_name_transform::Union{Function,Nothing}  # Transform model names for this provider
    schema::AbstractRequestSchema     # Request schema for this provider
    notes::String
end

const PROVIDER_INFO = Dict{String,ProviderInfo}(
    # OpenAI-compatible / OpenAI-style providers
    "openai" => ProviderInfo(
        "https://api.openai.com/v1",
        "Bearer",
        "OPENAI_API_KEY",
        Dict{String,String}(),
        openai_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI API"),
    "mistral" => ProviderInfo(
        "https://api.mistral.ai/v1",
        "Bearer",
        "MISTRAL_API_KEY",
        Dict{String,String}(),
        mistral_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "fireworks" => ProviderInfo(
        "https://api.fireworks.ai/inference/v1",
        "Bearer",
        "FIREWORKS_API_KEY",
        Dict{String,String}(),
        fireworks_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "together" => ProviderInfo(
        "https://api.together.xyz/v1",
        "Bearer",
        "TOGETHER_API_KEY",
        Dict{String,String}(),
        together_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "groq" => ProviderInfo(
        "https://api.groq.com/openai/v1",
        "Bearer",
        "GROQ_API_KEY",
        Dict{String,String}(),
        groq_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "deepseek" => ProviderInfo(
        "https://api.deepseek.com/v1",
        "Bearer",
        "DEEPSEEK_API_KEY",
        Dict{String,String}(),
        deepseek_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "cerebras" => ProviderInfo(
        "https://api.cerebras.ai/v1",
        "Bearer",
        "CEREBRAS_API_KEY",
        Dict{String,String}(),
        cerebras_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "sambanova" => ProviderInfo(
        "https://api.sambanova.ai/v1",
        "Bearer",
        "SAMBANOVA_API_KEY",
        Dict{String,String}(),
        sambanova_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "xai" => ProviderInfo(
        "https://api.x.ai/v1",
        "Bearer",
        "XAI_API_KEY",
        Dict{String,String}(),
        xai_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "moonshotai" => ProviderInfo(
        "https://api.moonshot.ai/v1",
        "Bearer",
        "MOONSHOT_API_KEY",
        Dict{String,String}(),
        moonshotai_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "minimax" => ProviderInfo(
        "https://api.minimaxi.chat/v1",
        "Bearer",
        "MINIMAX_API_KEY",
        Dict{String,String}(),
        minimax_model_transform,
        ChatCompletionSchema(),
        "Uses custom endpoint paths for different model types"),

    # Anthropic-style, Cohere, Google AI Studio, etc.
    "anthropic" => ProviderInfo(
        "https://api.anthropic.com",
        "x-api-key",
        "ANTHROPIC_API_KEY",
        Dict("anthropic-version" => "2023-06-01"),
        anthropic_model_transform,
        AnthropicSchema(),
        "Uses Anthropic's native API format"),
    "cohere" => ProviderInfo(
        "https://api.cohere.ai/v1",
        "Bearer",
        "COHERE_API_KEY",
        Dict{String,String}(),
        cohere_model_transform,
        ChatCompletionSchema(),  # Assuming Cohere uses OpenAI-compatible format
        "Uses Cohere's native API format"),
    "google-ai-studio" => ProviderInfo(
        "https://generativelanguage.googleapis.com/v1beta",
        "x-goog-api-key",
        "GOOGLE_API_KEY",
        Dict{String,String}(),
        google_model_transform,
        GeminiSchema(),
        "Uses Google's Gemini API format, not OpenAI-compatible"),

    # Other OpenAI-compatible hosts
    "deepinfra" => ProviderInfo(
        "https://api.deepinfra.com/v1/openai",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API; env var name not standardized here"),
    "hyperbolic" => ProviderInfo(
        "https://api.hyperbolic.xyz/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API; env var name not standardized here"),
    "novita" => ProviderInfo(
        "https://api.novita.ai/v3/openai",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API; env var name not standardized here"),
    "siliconflow" => ProviderInfo(
        "https://api.siliconflow.cn/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API; env var name not standardized here"),
    "nvidia" => ProviderInfo(
        "https://integrate.api.nvidia.com/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API; env var name not standardized here"),
    "perplexity" => ProviderInfo(
        "https://api.perplexity.ai",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API; env var name not standardized here"),

    # Azure is special; host is resource-dependent
    "azure" => ProviderInfo(
        "https://<resource>.openai.azure.com",
        "api-key",
        "AZURE_OPENAI_API_KEY",
        Dict("api-version" => "2023-03-15-preview"),
        nothing,
        ChatCompletionSchema(),
        "Requires resource-specific host (AZURE_OPENAI_HOST) and deployment-based paths"),

    # Local / native providers
    "ollama" => ProviderInfo(
        "http://localhost:11434/v1",
        "Bearer",                     # Ignored; no auth_header needed when api_key is empty
        nothing,                      # No env var; api_key optional
        Dict{String,String}(),
        ollama_model_transform,
        ChatCompletionSchema(),
        "Local Ollama instance using OpenAI-compatible /v1/chat/completions API"),
)

"""
    add_provider(name::String, base_url::String, auth_header_format::String="Bearer", 
                api_key_env_var::Union{String,Nothing}=nothing, 
                default_headers::Dict{String,String}=Dict{String,String}(),
                model_name_transform::Union{Function,Nothing}=nothing,
                schema::AbstractRequestSchema=ChatCompletionSchema(),
                notes::String="Custom provider")

Add a provider to the registry.

# Example
```julia
add_provider("echo", "http://localhost:8080/v1", "Bearer", "ECHO_API_KEY")
add_provider("ollama", "http://localhost:11434/v1", "Bearer")
```
"""
function add_provider(name::String, base_url::String,
                     auth_header_format::String="Bearer",
                     api_key_env_var::Union{String,Nothing}=nothing,
                     default_headers::Dict{String,String}=Dict{String,String}(),
                     model_name_transform::Union{Function,Nothing}=nothing,
                     schema::AbstractRequestSchema=ChatCompletionSchema(),
                     notes::String="Custom provider")
    PROVIDER_INFO[lowercase(name)] = ProviderInfo(
        base_url, auth_header_format, api_key_env_var, default_headers, model_name_transform, schema, notes
    )
end

"""
Transform model name according to provider-specific rules.
"""
function transform_model_name(provider_info::ProviderInfo, model_id::AbstractString)
    return provider_info.model_name_transform === nothing ? model_id : provider_info.model_name_transform(model_id)
end

"""
    remove_provider(name::String)

Remove a provider from the registry.
"""
function remove_provider(name::String)
    delete!(PROVIDER_INFO, lowercase(name))
end

"""
    list_providers()

List all registered providers.
"""
function list_providers()
    return collect(keys(PROVIDER_INFO))
end

"""
Get the provider info for a given slug, or `nothing` if unknown.
"""
function get_provider_info(provider_slug::AbstractString)::Union{ProviderInfo,Nothing}
    provider_slug = resolve_model_alias(provider_slug)
    return get(PROVIDER_INFO, lowercase(provider_slug), nothing)
end

"Get just the base URL for a provider slug, or `nothing` if unknown."
function get_provider_base_url(provider_slug::AbstractString)::Union{String,Nothing}
    info = get_provider_info(provider_slug)
    return info === nothing ? nothing : info.base_url
end

"Build an auth header pair (name => value) for a provider + API key, or `nothing` if provider is unknown."
function get_provider_auth_header(provider_slug::AbstractString,
        api_key::AbstractString)::Union{Pair{String,String},Nothing}
    info = get_provider_info(provider_slug)
    info === nothing && return nothing
    return get_provider_auth_header(info, api_key)
end

"Build an auth header pair (name => value) for a ProviderInfo + API key."
function get_provider_auth_header(info::ProviderInfo, api_key::AbstractString)::Pair{String,String}
    return if info.auth_header_format == "Bearer"
        "Authorization" => "Bearer $api_key"
    elseif info.auth_header_format == "x-api-key"
        "x-api-key" => api_key
    elseif info.auth_header_format == "api-key"
        "api-key" => api_key
    elseif info.auth_header_format == "x-goog-api-key"
        "x-goog-api-key" => api_key
    else
        "Authorization" => "$(info.auth_header_format) $api_key"
    end
end

"""
Build complete headers for a provider request.
"""
function build_headers(provider_info::ProviderInfo, api_key::AbstractString)
    headers = ["Content-Type" => "application/json"]
    
    # Add auth header only if provider expects one and api_key is non-empty
    if !isempty(api_key) && provider_info.auth_header_format != ""
        auth_header = get_provider_auth_header(provider_info, api_key)
        push!(headers, auth_header)
    end
    
    # Add default headers for this provider
    for (k, v) in provider_info.default_headers
        push!(headers, k => v)
    end
    
    return headers
end

"Return the configured API key env var name for a provider, or `nothing`."
function get_provider_env_var_name(provider_slug::AbstractString)::Union{String,Nothing}
    info = get_provider_info(provider_slug)
    return info === nothing ? nothing : info.api_key_env_var
end

"List all known provider slugs."
function list_known_providers()::Vector{String}
    return collect(keys(PROVIDER_INFO))
end

"Check if this provider slug is known."
function is_known_provider(provider_slug::AbstractString)::Bool
    return haskey(PROVIDER_INFO, provider_slug)
end

"""
    extract_provider_from_model(model_name::String) -> String

Extract provider name from model name in format "provider:author/model_id" or fallback to "openai".

# Examples
```julia
extract_provider_from_model("openai:openai/gpt-4") # => "openai"
extract_provider_from_model("anthropic:anthropic/claude-3-5-sonnet") # => "anthropic"
extract_provider_from_model("cerebras:meta-llama/llama-3.1-8b") # => "cerebras"
extract_provider_from_model("gpt-4") # => "openai" (fallback)
```
"""
function extract_provider_from_model(model_name::String)
    model_name = resolve_model_alias(model_name)
    colon_idx = findfirst(':', model_name)
    if colon_idx !== nothing
        return lowercase(model_name[1:colon_idx-1])
    end
    @warn "Provider was missing from slug: $model_name"
    return "openai"
end

"""
Calculate cost for a given endpoint and token usage.
Unwraps `.pricing`. Warns if cost cannot be determined (e.g. missing pricing).
"""
function calculate_cost(endpoint::ProviderEndpoint, tokens::Union{Nothing,Dict})
    if endpoint.pricing === nothing
        @warn "No pricing available on endpoint; cannot calculate cost." endpoint=endpoint tokens=tokens
        return nothing
    end

    cost = calculate_cost(endpoint.pricing, tokens)

    if cost === nothing
        @warn "Pricing present but resulted in zero/undefined cost; check pricing fields and tokens." endpoint=endpoint tokens=tokens
    end

    return cost
end

"""
Get the appropriate schema for a provider info and model.
For OpenAI, use ResponseSchema for gpt-5 and o-series models.
"""
function get_provider_schema(provider_info::ProviderInfo, model_id::AbstractString)::AbstractRequestSchema
    # Special case: OpenAI's gpt-5 and o-series use Response API
    if provider_info.schema isa ChatCompletionSchema && 
       (startswith(model_id, "gpt-5") || startswith(model_id, "o1-") || startswith(model_id, "o3-"))
        return ResponseSchema()
    end
    return provider_info.schema
end


