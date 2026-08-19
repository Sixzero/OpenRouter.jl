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
    _list_models_unfiltered(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}

Internal helper: return all models without any provider-based filtering, using the raw API.
Used by update_db to avoid recursive use of the cache.
"""
function _list_models_unfiltered(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}
    json_str = list_models_raw(api_key)
    return parse_models(json_str)
end

"""
    list_models(provider_filter::Union{String, Nothing} = nothing, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}

Return parsed model list as Julia structs, optionally filtered by provider.
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_models(provider_filter::Union{String, Nothing} = nothing, api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterModel}
    if provider_filter === nothing
        # No filtering needed, use the fast path
        return _list_models_unfiltered(api_key)
    end

    # For provider filtering, we need to check endpoints
    # Use the cache system to get models with endpoint information
    cache = get_global_cache()

    # If cache is empty or we need endpoint data, update it
    if isempty(cache.models)
        cache = update_db(api_key=api_key, fetch_endpoints=true)
    else
        # Check if we have endpoint data for models, if not fetch it
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

"""
    list_embeddings_models_raw(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String

Return raw JSON string of embedding models list.
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_embeddings_models_raw(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::String
    return read(`curl -s -H "Authorization: Bearer $api_key" https://openrouter.ai/api/v1/embeddings/models`, String)
end

"""
    list_embeddings_models(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterEmbeddingModel}

Return parsed embedding models list as Julia structs.
Uses OPENROUTER_API_KEY environment variable by default.
"""
function list_embeddings_models(api_key::String = get(ENV, "OPENROUTER_API_KEY", ""))::Vector{OpenRouterEmbeddingModel}
    json_str = list_embeddings_models_raw(api_key)
    return parse_embedding_models(json_str)
end

"""
    request_size_report(body) -> String

Human-readable size summary of a serialized request body: total bytes and, when the
JSON parses, the per-part breakdown (system, tools, each message) sorted biggest-first.
Used to diagnose "max request size" rejections — you see WHICH part blew the budget,
not just that the whole thing was too big.
"""
function request_size_report(body::AbstractString; top::Int=5)::String
    total = sizeof(body)
    parts = String[]
    try
        payload = JSON3.read(body)
        sized = Tuple{String,Int}[]
        for k in (:system, :system_instruction, :tools, :instructions)
            haskey(payload, k) && push!(sized, (String(k), sizeof(JSON3.write(payload[k]))))
        end
        msgs = haskey(payload, :messages) ? payload[:messages] :
               haskey(payload, :contents) ? payload[:contents] :
               haskey(payload, :input) ? payload[:input] : nothing
        if msgs !== nothing
            for (i, m) in enumerate(msgs)
                role = m isa JSON3.Object && haskey(m, :role) ? String(m[:role]) : "msg"
                push!(sized, ("$(role)[$i]", sizeof(JSON3.write(m))))
            end
        end
        sort!(sized, by=last, rev=true)
        parts = ["$(n)=$(_fmt_bytes(b))" for (n, b) in first(sized, top)]
    catch
        # Non-JSON or unexpected shape: total size is still the useful signal.
    end
    return isempty(parts) ? "total=$(_fmt_bytes(total))" :
           "total=$(_fmt_bytes(total)), top: " * join(parts, ", ")
end

_fmt_bytes(b::Integer) = b < 1024 ? "$(b)B" :
                         b < 1024^2 ? string(round(b / 1024, digits=1), "KB") :
                         string(round(b / 1024^2, digits=2), "MB")

"""
    log_request_failure(status, body)

Log a failed request's size breakdown plus a body snippet. Covers every 4xx/5xx (413
"payload too large", 400 "max request size exceeded", provider-specific variants), so
oversized-request failures are diagnosable without re-running with `verbose=true`.
"""
function log_request_failure(status::Integer, body::AbstractString; report::AbstractString=request_size_report(body))
    @error "API $(status): request payload details" size=report body_snippet=_snippet(body)
end

"""
    _is_size_error(status, response_body) -> Bool

Whether the failure is plausibly caused by request size, i.e. whether the size breakdown
is worth showing to whoever sees the error. 413 always is; 400 only when the provider says
so. A 429 (rate limit / provider capacity) never is — the payload was fine.
"""
_is_size_error(status::Integer, response_body::AbstractString) =
    status == 413 || (status == 400 &&
        occursin(r"too large|too long|max.{0,20}size|exceed.{0,20}(size|length)|context length"i, response_body))

# String-index-safe prefix (payloads contain multibyte UTF-8; `body[1:500]` can throw).
_snippet(s::AbstractString, n::Int=500) = sizeof(s) <= n ? String(s) : String(first(s, n))
