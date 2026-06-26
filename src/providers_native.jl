using HTTP
using JSON3

"""
    list_native_models(provider_slug::String, api_key::String = "")::Vector{Dict}

List models using a provider's native API.
Returns raw model data as returned by the provider.

# Example
```julia
models = list_native_models("cerebras")
models = list_native_models("openai", "your-api-key")
```
"""
function list_native_models(provider_slug::String, api_key::String = "")::Vector{Dict}
    provider_info = get_provider_info(provider_slug)
    provider_info === nothing && throw(ArgumentError("Unknown provider: $provider_slug"))
    
    # Get API key from env if not provided. Providers without a configured env var
    # (e.g. local Ollama) need no auth, so an empty key is fine.
    if isempty(api_key)
        env_var = provider_info.api_key_env_var
        if env_var !== nothing
            api_key = get(ENV, env_var, "")
            isempty(api_key) && throw(ArgumentError("API key not found in environment variable $env_var"))
        end
    end
    
    return fetch_native_models(provider_info, api_key)
end

"""
    fetch_native_models(provider_info::ProviderInfo, api_key::String)::Vector{Dict}

Fetch models directly from a provider's native API.
Returns raw model data as returned by the provider.
"""
function fetch_native_models(provider_info::ProviderInfo, api_key::String)::Vector{Dict}
    headers = build_headers(provider_info, api_key)
    url = build_native_models_url(provider_info)
    
    response = HTTP.get(url, headers)
    response.status != 200 && error("Provider API request failed with status $(response.status): $(String(response.body))")
    
    data = JSON3.read(response.body, Dict)
    # Ollama's /api/tags returns {"models": [...]}; OpenAI-compatible APIs return {"data": [...]}.
    return get(data, provider_info.schema isa OllamaSchema ? "models" : "data", [])
end

"""
    build_native_models_url(provider_info::ProviderInfo)::String

Build the models endpoint URL for a provider's native API.
"""
function build_native_models_url(provider_info::ProviderInfo)::String
    base = provider_info.base_url
    
    # Ollama lists models at /api/tags (base already ends in /api)
    if provider_info.schema isa OllamaSchema
        return "$base/tags"
    # Handle provider-specific endpoint paths
    elseif endswith(base, "/v1")
        # Base URL already includes version (e.g., most OpenAI-compatible APIs)
        return "$base/models"
    else
        # Base URL needs version added (e.g., Anthropic)
        return "$base/v1/models"
    end
end