#!/usr/bin/env julia

using OpenRouter
using JSON3

# ---------- Configuration ----------

const EXCLUDED_PROVIDERS = Set([
    "google",
    "amazon-bedrock"
])

# ---------- Helpers ----------

"Convert an OpenRouter Pricing struct into a Dict with numeric fields."
function pricing_to_numeric_dict(p::Pricing)
    # OpenRouter exposes prices mostly as strings (e.g. "0.000005").
    # Frontend expects numbers in USD / token.
    parse_or_nothing(x) = x === nothing ? nothing : parse(Float64, String(x))

    raw = Dict(
        "prompt" => parse_or_nothing(p.prompt),
        "completion" => parse_or_nothing(p.completion),
        "request" => parse_or_nothing(p.request),
        "image" => parse_or_nothing(p.image),
        "web_search" => parse_or_nothing(p.web_search),
        "internal_reasoning" => parse_or_nothing(p.internal_reasoning),
        "image_output" => parse_or_nothing(p.image_output),
        "audio" => parse_or_nothing(p.audio),
        "input_audio_cache" => parse_or_nothing(p.input_audio_cache),
        "input_cache_read" => parse_or_nothing(p.input_cache_read),
        "input_cache_write" => parse_or_nothing(p.input_cache_write),
        "discount" => p.discount
    )

    # Drop keys where value is `nothing`, sort keys for consistent ordering
    pricing = Dict{String,Any}()
    for k in sort(collect(keys(raw)))
        v = raw[k]
        v === nothing && continue
        pricing[k] = v
    end
    return pricing
end

"""
Build a simplified per-endpoint description suitable for frontend consumption.
"""
function endpoint_to_frontend_dict(ep::ProviderEndpoint)
    return Dict(
        "provider_name" => replace(ep.provider_name, " " => "-"),
        "endpoint_name" => ep.name,
        "context_length" => ep.context_length,
        "max_completion_tokens" => ep.max_completion_tokens,
        "pricing" => pricing_to_numeric_dict(ep.pricing),
        "tag" => ep.tag,
        "quantization" => ep.quantization,
        "supports_implicit_caching" => ep.supports_implicit_caching,
        "status" => ep.status
    )
end

"""
From a list of endpoints, derive a single 'primary' prompt / completion price
to fill legacy fields cost_of_token_prompt / cost_of_token_generation.

Strategy:
- Prefer endpoints with non-nothing prompt and completion pricing.
- Among them, pick the one with the lowest prompt price.
"""
function pick_primary_prices(endpoints::Vector{ProviderEndpoint})
    best_prompt = nothing
    best_completion = nothing

    for ep in endpoints
        p = ep.pricing
        p_prompt = p.prompt === nothing ? nothing : parse(Float64, String(p.prompt))
        p_comp = p.completion === nothing ? nothing : parse(Float64, String(p.completion))

        if p_prompt === nothing || p_comp === nothing
            continue
        end

        if best_prompt === nothing || p_prompt < best_prompt
            best_prompt = p_prompt
            best_completion = p_comp
        end
    end

    return best_prompt, best_completion
end

"""
Check if an endpoint should be excluded based on its provider.
"""
function should_exclude_endpoint(ep::ProviderEndpoint)
    provider_normalized = lowercase(replace(ep.provider_name, " " => "-"))
    return provider_normalized in EXCLUDED_PROVIDERS || startswith(provider_normalized, "echo_")
end

"""
Process a single model and return its spec dict and excluded count.
"""
function process_model(m::OpenRouterModel, i::Int, total::Int)
    println("[$i/$total] Processing model $(m.id)")

    ep_dicts = Any[]
    endpoints = ProviderEndpoint[]
    excluded_count = 0

    try
        eps = list_endpoints(m.id)
        all_endpoints = eps.endpoints

        # Filter out endpoints from excluded providers
        filtered_endpoints = filter(ep -> !should_exclude_endpoint(ep), all_endpoints)
        excluded_count = length(all_endpoints) - length(filtered_endpoints)

        if excluded_count > 0
            println("  Filtered out $excluded_count endpoint(s) from excluded providers")
        end

        endpoints = filtered_endpoints
        # Sort endpoints by tag for consistent ordering
        sorted_endpoints = sort(filtered_endpoints, by=ep -> ep.tag)
        ep_dicts = [endpoint_to_frontend_dict(ep) for ep in sorted_endpoints]
    catch err
        @warn "Failed to fetch endpoints for model" id=m.id exception=(err, catch_backtrace())
    end

    primary_prompt, primary_completion = pick_primary_prices(endpoints)

    spec = Dict(
        "id" => m.id,
        "name" => m.name,
        # "description" => m.description === nothing ? "" : m.description,
        # "context_length" => m.context_length,
        "created" => m.created,
        "endpoints" => ep_dicts
    )

    return spec, excluded_count
end

"""
Export all models + endpoints from OpenRouter into a JSON structure
compatible with the frontend's ModelsData type (plus extra endpoint details).
"""
function build_models_data()
    println("Fetching models from OpenRouter...")
    models = list_models()
    println("Fetched $(length(models)) models")

    # Process models in parallel with 8 concurrent tasks
    total = length(models)
    results = asyncmap(enumerate(models); ntasks=8) do (i, m)
        process_model(m, i, total)
    end

    specs = [r[1] for r in results]
    excluded_endpoints_count = sum(r[2] for r in results)

    println("\nTotal excluded endpoints: $excluded_endpoints_count")

    # Sort models alphabetically by id for consistent ordering
    sort!(specs, by=d -> d["id"])

    aliases = Dict{String,String}()

    # Providers list sorted alphabetically for consistent ordering
    providers = sort(list_providers())

    return Dict(
        "models" => specs,
        "aliases" => aliases,
        "total_models" => length(specs),
        "total_aliases" => length(aliases),
        "providers" => providers,
        "excluded_providers" => sort(collect(EXCLUDED_PROVIDERS))
    )
end

# ---------- Main ----------

function main()
    # Priority: CLI arg > env var > default
    out_file = if length(ARGS) >= 1
        ARGS[1]
    elseif haskey(ENV, "OPENROUTER_MODELS_OUTPUT")
        ENV["OPENROUTER_MODELS_OUTPUT"]
    else
        "models_data_openrouter.json"
    end

    data = build_models_data()

    open(out_file, "w") do io
        JSON3.pretty(io, data)
    end

    total_models = data["total_models"]
    println("✅ Exported $total_models models to $out_file")
end

main()