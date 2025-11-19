using JSON3

"""
    list_models_raw(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String

Return raw JSON string of model list.
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_models_raw(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String
    return read(`curl -s -H "Authorization: Bearer $api_key" https://openrouter.ai/api/v1/models`, String)
end

"""
    list_models(provider_filter::Union{String, Nothing} = nothing, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}

Return parsed model list as Julia structs, optionally filtered by provider.
Uses OPENROUTER_API_KEY environment variable by default.

# Arguments
- `provider_filter::Union{String, Nothing}`: Optional provider name to filter by (e.g., "anthropic", "openai", "groq")
- `api_key::String`: API key for OpenRouter
"""
function list_models(provider_filter::Union{String, Nothing} = nothing, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}
    if provider_filter === nothing
        # No filtering needed, use the fast path
        json_str = list_models_raw(api_key)
        return parse_models(json_str)
    end
    
    # For provider filtering, we need to check endpoints
    # Use the cache system to get models with endpoint information
    cache = get_global_cache()
    
    # If cache is empty or we need endpoint data, update it
    if isempty(cache.models)
        cache = update_db(api_key=api_key, fetch_endpoints=true)
    else
        # Check if we have endpoint data for models, if not fetch it
        # was: models_without_endpoints = sum(1 for cached in values(cache.models) if !cached.endpoints_fetched)
        needs_endpoints = any(!cached.endpoints_fetched for cached in values(cache.models))
        if needs_endpoints
            cache = update_db(api_key=api_key, fetch_endpoints=true, full_refresh=false)
        end
    end
    
    # Filter models by provider endpoints
    provider_lower = lowercase(provider_filter)
    filtered_models = OpenRouterModel[]
    
    for cached_model in values(cache.models)
        if cached_model.endpoints !== nothing
            # Check if any endpoint is hosted by the requested provider
            has_provider = any(cached_model.endpoints.endpoints) do endpoint
                lowercase(endpoint.provider_name) == provider_lower
            end
            
            if has_provider
                push!(filtered_models, cached_model.model)
            end
        end
    end
    
    return filtered_models
end

"""
    list_provider_endpoints(provider_filter::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{ProviderEndpoint}

Return all ProviderEndpoint entries hosted by the given provider.

This uses the cached model database with endpoints; it will fetch endpoints as needed
on first call.

Example:
```julia
groq_eps = list_provider_endpoints("groq")
for ep in groq_eps
    println(ep.provider_name, " ", ep.name, " (", ep.model_name, ")")
end
```
"""
function list_provider_endpoints(provider_filter::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{ProviderEndpoint}
    cache = get_global_cache()
    
    if isempty(cache.models)
        cache = update_db(api_key=api_key, fetch_endpoints=true)
    else
        needs_endpoints = any(!cached.endpoints_fetched for cached in values(cache.models))
        if needs_endpoints
            cache = update_db(api_key=api_key, fetch_endpoints=true, full_refresh=false)
        end
    end
    
    provider_lower = lowercase(provider_filter)
    endpoints = ProviderEndpoint[]
    
    for cached_model in values(cache.models)
        if cached_model.endpoints !== nothing
            for ep in cached_model.endpoints.endpoints
                if lowercase(ep.provider_name) == provider_lower
                    push!(endpoints, ep)
                end
            end
        end
    end
    
    return endpoints
end

"""
    list_providers_raw(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String

Return raw JSON string of providers for a specific model.
...
"""
function list_providers_raw(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String
    return read(`curl -s -H "Authorization: Bearer $api_key" https://openrouter.ai/api/v1/models/$model_id/endpoints`, String)
end

"""
    list_providers(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::ModelProviders

Return parsed providers for a specific model as Julia struct.
Model ID should be in format "author/slug" (e.g., "moonshotai/kimi-k2-thinking").
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_providers(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::ModelProviders
    json_str = list_providers_raw(model_id, api_key)
    return parse_endpoints(json_str)
end

"""
    list_endpoints_raw(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String

Return raw JSON string of endpoints for a specific model from OpenRouter.
Model ID should be in format "author/slug" (e.g., "moonshotai/kimi-k2-thinking").
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_endpoints_raw(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String
    return read(`curl -s -H "Authorization: Bearer $api_key" https://openrouter.ai/api/v1/models/$model_id/endpoints`, String)
end

"""
    list_endpoints(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::ModelProviders

Return parsed endpoints for a specific model from OpenRouter as Julia struct.
Model ID should be in format "author/slug" (e.g., "moonshotai/kimi-k2-thinking").
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_endpoints(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::ModelProviders
    json_str = list_endpoints_raw(model_id, api_key)
    return parse_endpoints(json_str)
end