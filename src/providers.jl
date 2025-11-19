
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

# Model name transformation functions
function anthropic_model_transform(model_id::AbstractString)
    # Replace dots with dashes in version numbers (e.g., "4.5" -> "4-5")
    return replace(model_id, r"(\d+)\.(\d+)" => s"\1-\2")
end

function google_model_transform(model_id::AbstractString)
    # Remove "google/" prefix if present
    return startswith(model_id, "google/") ? model_id[8:end] : model_id
end

const PROVIDER_INFO = Dict{String,ProviderInfo}(
    # OpenAI-compatible / OpenAI-style providers
    "openai" => ProviderInfo(
        "https://api.openai.com/v1",
        "Bearer",
        "OPENAI_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI API"),
    "mistral" => ProviderInfo(
        "https://api.mistral.ai/v1",
        "Bearer",
        "MISTRAL_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "fireworks" => ProviderInfo(
        "https://api.fireworks.ai/inference/v1",
        "Bearer",
        "FIREWORKS_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "together" => ProviderInfo(
        "https://api.together.xyz/v1",
        "Bearer",
        "TOGETHER_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "groq" => ProviderInfo(
        "https://api.groq.com/openai/v1",
        "Bearer",
        "GROQ_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "deepseek" => ProviderInfo(
        "https://api.deepseek.com/v1",
        "Bearer",
        "DEEPSEEK_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "cerebras" => ProviderInfo(
        "https://api.cerebras.ai/v1",
        "Bearer",
        "CEREBRAS_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "sambanova" => ProviderInfo(
        "https://api.sambanova.ai/v1",
        "Bearer",
        "SAMBANOVA_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "xai" => ProviderInfo(
        "https://api.x.ai/v1",
        "Bearer",
        "XAI_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "moonshotai" => ProviderInfo(
        "https://api.moonshot.ai/v1",
        "Bearer",
        "MOONSHOT_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "minimax" => ProviderInfo(
        "https://api.minimaxi.chat/v1",
        "Bearer",
        "MINIMAX_API_KEY",
        Dict{String,String}(),
        nothing,
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
        nothing,
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
        "Requires resource-specific host (AZURE_OPENAI_HOST) and deployment-based paths")
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
    auth_header = get_provider_auth_header(provider_info, api_key)
    
    headers = [
        auth_header,
        "Content-Type" => "application/json"
    ]
    
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
Get the appropriate schema for a provider info.
"""
get_provider_schema(provider_info::ProviderInfo)::AbstractRequestSchema = provider_info.schema