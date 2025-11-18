"""
Provider information for OpenRouter API providers.
Contains general information about each provider, not request-specific configuration.
"""

struct ProviderInfo
    base_url::String
    auth_header_format::String        # "Bearer", "Api-Key", "x-api-key", etc.
    api_key_env_var::Union{String,Nothing}  # Full env var name, e.g. "OPENAI_API_KEY"
    default_headers::Dict{String,String}    # Headers always sent with requests
    notes::String
end

const PROVIDER_INFO = Dict{String,ProviderInfo}(
    # OpenAI-compatible / OpenAI-style providers
    "openai" => ProviderInfo(
        "https://api.openai.com/v1",
        "Bearer",
        "OPENAI_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI API"),
    "mistral" => ProviderInfo(
        "https://api.mistral.ai/v1",
        "Bearer",
        "MISTRAL_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "fireworks" => ProviderInfo(
        "https://api.fireworks.ai/inference/v1",
        "Bearer",
        "FIREWORKS_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "together" => ProviderInfo(
        "https://api.together.xyz/v1",
        "Bearer",
        "TOGETHER_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "groq" => ProviderInfo(
        "https://api.groq.com/openai/v1",
        "Bearer",
        "GROQ_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "deepseek" => ProviderInfo(
        "https://api.deepseek.com/v1",
        "Bearer",
        "DEEPSEEK_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "cerebras" => ProviderInfo(
        "https://api.cerebras.ai/v1",
        "Bearer",
        "CEREBRAS_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "sambanova" => ProviderInfo(
        "https://api.sambanova.ai/v1",
        "Bearer",
        "SAMBANOVA_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "xai" => ProviderInfo(
        "https://api.x.ai/v1",
        "Bearer",
        "XAI_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "moonshotai" => ProviderInfo(
        "https://api.moonshot.ai/v1",
        "Bearer",
        "MOONSHOT_API_KEY",
        Dict{String,String}(),
        "Standard OpenAI-compatible API"),
    "minimax" => ProviderInfo(
        "https://api.minimaxi.chat/v1",
        "Bearer",
        "MINIMAX_API_KEY",
        Dict{String,String}(),
        "Uses custom endpoint paths for different model types"),

    # Anthropic-style, Cohere, Google AI Studio, etc.
    "anthropic" => ProviderInfo(
        "https://api.anthropic.com",
        "x-api-key",
        "ANTHROPIC_API_KEY",
        Dict("anthropic-version" => "2023-06-01"),
        "Uses Anthropic's native API format"),
    "cohere" => ProviderInfo(
        "https://api.cohere.ai/v1",
        "Bearer",
        "COHERE_API_KEY",
        Dict{String,String}(),
        "Uses Cohere's native API format"),
    "google-ai-studio" => ProviderInfo(
        "https://generativelanguage.googleapis.com/v1beta",
        "Bearer",
        "GOOGLE_API_KEY",
        Dict{String,String}(),
        "Uses Google's Gemini API format, not OpenAI-compatible"),

    # Other OpenAI-compatible hosts
    "deepinfra" => ProviderInfo(
        "https://api.deepinfra.com/v1/openai",
        "Bearer",
        nothing,
        Dict{String,String}(),
        "OpenAI-compatible API; env var name not standardized here"),
    "hyperbolic" => ProviderInfo(
        "https://api.hyperbolic.xyz/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        "OpenAI-compatible API; env var name not standardized here"),
    "novita" => ProviderInfo(
        "https://api.novita.ai/v3/openai",
        "Bearer",
        nothing,
        Dict{String,String}(),
        "OpenAI-compatible API; env var name not standardized here"),
    "siliconflow" => ProviderInfo(
        "https://api.siliconflow.cn/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        "OpenAI-compatible API; env var name not standardized here"),
    "nvidia" => ProviderInfo(
        "https://integrate.api.nvidia.com/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        "OpenAI-compatible API; env var name not standardized here"),
    "perplexity" => ProviderInfo(
        "https://api.perplexity.ai",
        "Bearer",
        nothing,
        Dict{String,String}(),
        "OpenAI-compatible API; env var name not standardized here"),

    # Azure is special; host is resource-dependent
    "azure" => ProviderInfo(
        "https://<resource>.openai.azure.com",
        "api-key",
        "AZURE_OPENAI_API_KEY",
        Dict("api-version" => "2023-03-15-preview"),
        "Requires resource-specific host (AZURE_OPENAI_HOST) and deployment-based paths")
)

"Get the provider info for a given slug, or `nothing` if unknown."
function get_provider_info(provider_slug::String)::Union{ProviderInfo,Nothing}
    return get(PROVIDER_INFO, provider_slug, nothing)
end

"Get just the base URL for a provider slug, or `nothing` if unknown."
function get_provider_base_url(provider_slug::String)::Union{String,Nothing}
    info = get_provider_info(provider_slug)
    return info === nothing ? nothing : info.base_url
end

"Build an auth header pair (name => value) for a provider + API key, or `nothing` if provider is unknown."
function get_provider_auth_header(provider_slug::String,
        api_key::String)::Union{Pair{String,String},Nothing}
    info = get_provider_info(provider_slug)
    info === nothing && return nothing
    return get_provider_auth_header(info, api_key)
end

"Build an auth header pair (name => value) for a ProviderInfo + API key."
function get_provider_auth_header(info::ProviderInfo, api_key::String)::Pair{String,String}
    return if info.auth_header_format == "Bearer"
        "Authorization" => "Bearer $api_key"
    elseif info.auth_header_format == "x-api-key"
        "x-api-key" => api_key
    elseif info.auth_header_format == "api-key"
        "api-key" => api_key
    else
        "Authorization" => "$(info.auth_header_format) $api_key"
    end
end

"Return the configured API key env var name for a provider, or `nothing`."
function get_provider_env_var_name(provider_slug::String)::Union{String,Nothing}
    info = get_provider_info(provider_slug)
    return info === nothing ? nothing : info.api_key_env_var
end

"List all known provider slugs."
function list_known_providers()::Vector{String}
    return collect(keys(PROVIDER_INFO))
end

"Check if this provider slug is known."
function is_known_provider(provider_slug::String)::Bool
    return haskey(PROVIDER_INFO, provider_slug)
end