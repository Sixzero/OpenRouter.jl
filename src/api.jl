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
    list_models(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}

Return parsed model list as Julia structs.
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_models(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}
    json_str = list_models_raw(api_key)
    return parse_models(json_str)
end

"""
    list_providers_raw(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String

Return raw JSON string of providers for a specific model.
Model ID should be in format "author/slug" (e.g., "moonshotai/kimi-k2-thinking").
Uses OPENROUTER_API_KEY environment variable by default.
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
    list_endpoints(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::ModelEndpoints

Return parsed endpoints for a specific model from OpenRouter as Julia struct.
Model ID should be in format "author/slug" (e.g., "moonshotai/kimi-k2-thinking").
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_endpoints(model_id::String, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::ModelEndpoints
    json_str = list_endpoints_raw(model_id, api_key)
    return parse_endpoints(json_str)
end