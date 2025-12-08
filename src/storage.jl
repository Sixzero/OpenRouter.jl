using JSON3
using Dates
using Scratch

# Storage-specific types
mutable struct CachedModel
    model::OpenRouterModel
    endpoints::Union{ModelProviders, Nothing}
    last_updated::DateTime
    endpoints_fetched::Bool
end

mutable struct ModelCache
    models::Dict{String, CachedModel}
    last_full_refresh::DateTime
end

# Global cache variable
const GLOBAL_CACHE = Ref{Union{ModelCache, Nothing}}(nothing)

# Global aliases - mapping short forms to provider:model format
const MODEL_ALIASES = Dict{String, String}(
    "gemf" => "google-ai-studio:google/gemini-2.5-flash-preview-09-2025",
    "gemfl" => "google-ai-studio:google/gemini-2.5-flash-lite-preview-09-2025",
    "claude" => "anthropic:anthropic/claude-sonnet-4.5",
    "gpt5" => "openai:openai/gpt-5.1",
)

"""
    resolve_model_alias(model_id::String)::String

Resolve a model alias to the full provider:model format.
If the input is not an alias, returns it unchanged.

# Example
```julia
resolve_model_alias("gemf")  # Returns "google-ai-studio:google/gemini-2.5-flash-preview-09-2025"
resolve_model_alias("anthropic:claude-3-sonnet")  # Returns unchanged
```
"""
function resolve_model_alias(model_id::AbstractString)::AbstractString
    return get(MODEL_ALIASES, model_id, model_id)
end

"""
    list_aliases()::Dict{String, String}

List all available model aliases.
"""
function list_aliases()::Dict{String, String}
    return copy(MODEL_ALIASES)
end

get_cache_dir() = @get_scratch!("openrouter_cache")
get_cache_file() = joinpath(get_cache_dir(), "models.json")

# Serialization helpers
function model_to_dict(model::OpenRouterModel)
    return Dict(
        "id" => model.id,
        "name" => model.name,
        "description" => model.description,
        "context_length" => model.context_length,
        "pricing" => model.pricing === nothing ? nothing : pricing_to_dict(model.pricing),
        "architecture" => model.architecture === nothing ? nothing : architecture_to_dict(model.architecture),
        "created" => model.created
    )
end

function pricing_to_dict(pricing::Pricing)
    return Dict(
        "prompt" => pricing.prompt,
        "completion" => pricing.completion,
        "request" => pricing.request,
        "image" => pricing.image,
        "web_search" => pricing.web_search,
        "internal_reasoning" => pricing.internal_reasoning,
        "image_output" => pricing.image_output,
        "audio" => pricing.audio,
        "input_audio_cache" => pricing.input_audio_cache,
        "input_cache_read" => pricing.input_cache_read,
        "input_cache_write" => pricing.input_cache_write,
        "discount" => pricing.discount
    )
end

function architecture_to_dict(arch::Architecture)
    return Dict(
        "modality" => arch.modality,
        "input_modalities" => arch.input_modalities,
        "output_modalities" => arch.output_modalities,
        "tokenizer" => arch.tokenizer,
        "instruct_type" => arch.instruct_type
    )
end

function endpoints_to_dict(endpoints::ModelProviders)
    return Dict(
        "id" => endpoints.id,
        "name" => endpoints.name,
        "created" => endpoints.created,
        "description" => endpoints.description,
        "architecture" => endpoints.architecture === nothing ? nothing : architecture_to_dict(endpoints.architecture),
        "endpoints" => [endpoint_to_dict(ep) for ep in endpoints.endpoints]
    )
end

function endpoint_to_dict(endpoint::ProviderEndpoint)
    return Dict(
        "name" => endpoint.name,
        "model_name" => endpoint.model_name,
        "context_length" => endpoint.context_length,
        "pricing" => pricing_to_dict(endpoint.pricing),
        "provider_name" => endpoint.provider_name,
        "tag" => endpoint.tag,
        "quantization" => endpoint.quantization,
        "max_completion_tokens" => endpoint.max_completion_tokens,
        "max_prompt_tokens" => endpoint.max_prompt_tokens,
        "supported_parameters" => endpoint.supported_parameters,
        "uptime_last_30m" => endpoint.uptime_last_30m,
        "supports_implicit_caching" => endpoint.supports_implicit_caching,
        "status" => endpoint.status
    )
end

# Deserialization helpers
function dict_to_model(data)
    pricing = data["pricing"] === nothing ? nothing : dict_to_pricing(data["pricing"])
    architecture = data["architecture"] === nothing ? nothing : dict_to_architecture(data["architecture"])
    
    return OpenRouterModel(
        data["id"],
        data["name"],
        data["description"],
        data["context_length"],
        pricing,
        architecture,
        data["created"]
    )
end

function dict_to_pricing(data)
    return Pricing(
        data["prompt"],
        data["completion"],
        data["request"],
        data["image"],
        data["web_search"],
        data["internal_reasoning"],
        data["image_output"],
        data["audio"],
        data["input_audio_cache"],
        data["input_cache_read"],
        data["input_cache_write"],
        data["discount"]
    )
end

function dict_to_architecture(data)
    return Architecture(
        data["modality"],
        data["input_modalities"],
        data["output_modalities"],
        data["tokenizer"],
        data["instruct_type"]
    )
end

function dict_to_endpoints(data)
    architecture = data["architecture"] === nothing ? nothing : dict_to_architecture(data["architecture"])
    endpoints = [dict_to_endpoint(ep_data) for ep_data in data["endpoints"]]
    
    return ModelProviders(
        data["id"],
        data["name"],
        data["created"],
        data["description"],
        architecture,
        endpoints
    )
end

function dict_to_endpoint(data)
    pricing = dict_to_pricing(data["pricing"])
    
    return ProviderEndpoint(
        data["name"],
        data["model_name"],
        data["context_length"],
        pricing,
        data["provider_name"],
        data["tag"],
        data["quantization"],
        data["max_completion_tokens"],
        data["max_prompt_tokens"],
        data["supported_parameters"],
        data["uptime_last_30m"],
        data["supports_implicit_caching"],
        data["status"]
    )
end

function save_cache(cache::ModelCache)
    cache_file = get_cache_file()
    cache_data = Dict(
        "models" => Dict(
            id => Dict(
                "model" => model_to_dict(cached.model),
                "endpoints" => cached.endpoints === nothing ? nothing : endpoints_to_dict(cached.endpoints),
                "last_updated" => string(cached.last_updated),
                "endpoints_fetched" => cached.endpoints_fetched
            ) for (id, cached) in cache.models
        ),
        "last_full_refresh" => string(cache.last_full_refresh)
    )
    open(cache_file, "w") do f
        JSON3.pretty(f, cache_data)
    end
    GLOBAL_CACHE[] = cache
end

function load_cache()::ModelCache
    cache_file = get_cache_file()
    if !isfile(cache_file)
        return ModelCache(Dict{String, CachedModel}(), DateTime(0))
    end

    cache_data = JSON3.read(read(cache_file, String))
    models = Dict{String, CachedModel}()
    for (id, data) in cache_data.models
        id_str = string(id)  # Convert Symbol to String
        model = dict_to_model(data.model)
        endpoints = data.endpoints === nothing ? nothing : dict_to_endpoints(data.endpoints)
        last_updated = DateTime(data.last_updated)
        endpoints_fetched = data.endpoints_fetched
        models[id_str] = CachedModel(model, endpoints, last_updated, endpoints_fetched)
    end
    last_full_refresh = DateTime(cache_data.last_full_refresh)
    return ModelCache(models, last_full_refresh)
end

get_global_cache() = (GLOBAL_CACHE[] === nothing ? GLOBAL_CACHE[] = load_cache() : GLOBAL_CACHE[])

function update_db(;
    api_key::String = get(ENV, "OPENROUTER_API_KEY", ""),
    full_refresh::Bool = false,
    fetch_endpoints::Bool = true
)::ModelCache
    @info "Downloading new models. Updating db..." full_refresh fetch_endpoints
    cache = get_global_cache()
    now_time = now()

    # Use internal unfiltered list to avoid recursion through provider-filtering list_models
    models = _list_models_unfiltered(api_key)
    for model in models
        if full_refresh || !haskey(cache.models, model.id)
            cache.models[model.id] = CachedModel(model, nothing, now_time, false)
        else
            existing = cache.models[model.id]
            cache.models[model.id] = CachedModel(model, existing.endpoints, now_time, existing.endpoints_fetched)
        end
    end

    if fetch_endpoints
        for (model_id, cached_model) in cache.models
            if full_refresh || !cached_model.endpoints_fetched
                try
                    endpoints = list_endpoints(model_id, api_key)
                    cache.models[model_id] = CachedModel(cached_model.model, endpoints, now_time, true)
                catch
                    cache.models[model_id] = CachedModel(cached_model.model, nothing, now_time, true)
                end
            end
        end
    end

    cache.last_full_refresh = now_time
    save_cache(cache)
    return cache
end

function get_model(
    model_id::AbstractString;
    api_key::String = get(ENV, "OPENROUTER_API_KEY", ""),
    fetch_endpoints::Bool = false
)::Union{CachedModel, Nothing}
    cache = get_global_cache()

    if isempty(cache.models)
        cache = update_db(api_key = api_key, fetch_endpoints = fetch_endpoints)
    end

    if haskey(cache.models, model_id)
        cached = cache.models[model_id]
        if fetch_endpoints && !cached.endpoints_fetched
            cache = update_db(api_key = api_key, full_refresh = false, fetch_endpoints = true)
            return get(cache.models, model_id, nothing)
        end
        return cached
    end

    cache = update_db(api_key = api_key, fetch_endpoints = fetch_endpoints, full_refresh=false)
    return get(cache.models, model_id, nothing)
end

function list_cached_models()::Vector{OpenRouterModel}
    cache = get_global_cache()
    return [cached.model for cached in values(cache.models)]
end

function search_models(query::String; case_sensitive::Bool = false)::Vector{CachedModel}
    cache = get_global_cache()
    
    if isempty(cache.models)
        cache = update_db(fetch_endpoints=false)
    end
    
    query_processed = case_sensitive ? query : lowercase(query)
    
    results = CachedModel[]
    for cached in values(cache.models)
        model_id = case_sensitive ? cached.model.id : lowercase(cached.model.id)
        model_name = case_sensitive ? cached.model.name : lowercase(cached.model.name)
        
        if contains(model_id, query_processed) || contains(model_name, query_processed)
            push!(results, cached)
        end
    end
    
    return results
end

"""
    add_custom_model(id::String, name::String, description::String="Custom model",
                    context_length::Union{Int,Nothing}=nothing,
                    pricing::Union{Pricing,Nothing}=nothing,
                    architecture::Union{Architecture,Nothing}=nothing)

Add a custom model to the local cache.

# Example
```julia
add_custom_model("echo/100tps", "Echo 100 TPS", "Fast echo model for testing", 8192)
add_custom_model("local/llama3", "Local Llama 3", "Self-hosted Llama 3", 4096)
```
"""
function add_custom_model(id::String, name::String, description::String="Custom model",
                         context_length::Union{Int,Nothing}=nothing,
                         pricing::Union{Pricing,Nothing}=nothing,
                         architecture::Union{Architecture,Nothing}=nothing)
    cache = get_global_cache()
    
    # Create default pricing if not provided
    if pricing === nothing
        pricing = Pricing("0", "0", "0", "0", "0", "0", nothing, nothing, nothing, "0", nothing, nothing)
    end
    
    # Create default architecture if not provided
    if architecture === nothing
        architecture = Architecture("text->text", ["text"], ["text"], "custom", nothing)
    end
    
    model = OpenRouterModel(id, name, description, context_length, pricing, architecture, trunc(Int, time()))
    cached_model = CachedModel(model, nothing, now(), false)
    
    cache.models[id] = cached_model
    save_cache(cache)
    
    return cached_model
end

"""
    remove_custom_model(id::String)

Remove a custom model from the local cache.
"""
function remove_custom_model(id::String)
    cache = get_global_cache()
    if haskey(cache.models, id)
        delete!(cache.models, id)
        save_cache(cache)
        return true
    end
    return false
end

"""
    add_model(id::String, name::String, description::String="Custom model",
             context_length::Union{Int,Nothing}=nothing,
             pricing::Union{Pricing,Nothing}=nothing,
             architecture::Union{Architecture,Nothing}=nothing)

Add a model to the local cache.

# Example
```julia
add_model("echo/100tps", "Echo 100 TPS", "Fast echo model for testing", 8192)
add_model("ollama/llama3", "Local Llama 3", "Self-hosted Llama 3", 4096)
```
"""
function add_model(id::String, name::String, description::String="Custom model",
                  context_length::Union{Int,Nothing}=nothing,
                  pricing::Union{Pricing,Nothing}=nothing,
                  architecture::Union{Architecture,Nothing}=nothing)
    cache = get_global_cache()
    
    # Create default pricing if not provided
    if pricing === nothing
        pricing = Pricing("0", "0", "0", "0", "0", "0", nothing, nothing, nothing, "0", nothing, nothing)
    end
    
    # Create default architecture if not provided
    if architecture === nothing
        architecture = Architecture("text->text", ["text"], ["text"], "custom", nothing)
    end
    
    model = OpenRouterModel(id, name, description, context_length, pricing, architecture, trunc(Int, time()))
    cached_model = CachedModel(model, nothing, now(), false)
    
    cache.models[id] = cached_model
    save_cache(cache)
    
    return cached_model
end

"""
    remove_model(id::String)

Remove a model from the local cache.
"""
function remove_model(id::String)
    cache = get_global_cache()
    if haskey(cache.models, id)
        delete!(cache.models, id)
        save_cache(cache)
        return true
    end
    return false
end