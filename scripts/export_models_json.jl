#!/usr/bin/env julia

using OpenRouter
using JSON3
using Dates

# ---------- Configuration ----------

const EXCLUDED_PROVIDERS = Set([
    "amazon-bedrock",
    "azure",
    "cloudflare",
    "friendli",
    "google",
    "io-net",
    # bulk dropped: providers not in PROVIDER_INFO (no native routing)
    "ai21",
    "aionlabs",
    "akashml",
    "alibaba",
    "ambient",
    "arcee-ai",
    "baidu",
    "baseten",
    "darkbloom",
    "decart",
    "dekallm",
    "digitalocean",
    "gmicloud",
    "inception",
    "inceptron",
    "infermatic",
    "inflection",
    "ionstream",
    "liquid",
    "mancer-2",
    "mara",
    "modelrun",
    "morph",
    "nebius",
    "nex-agi",
    "nextbit",
    "openinference",
    "parasail",
    "perceptron",
    "phala",
    "reka",
    "relace",
    "seed",
    "stepfun",
    "streamlake",
    "switchpoint",
    "upstage",
    "venice",
    "wafer",
    "wandb",
])

# (provider, model_id) combos that OpenRouter advertises but that are not actually
# reachable on that provider's endpoint (model id mismatch, not on account, etc.).
# Provider is matched normalized (lowercase, spaces->hyphens); model_id lowercased.
const EXCLUDED_ENDPOINTS = Set([
    ("cerebras", "qwen/qwen3-32b"),
    ("together", "meta-llama/llama-4-scout"),
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
Check if an endpoint should be excluded based on its provider, or because the
(provider, model_id) combo is listed in EXCLUDED_ENDPOINTS.
"""
function should_exclude_endpoint(ep::ProviderEndpoint, model_id::AbstractString)
    provider_normalized = lowercase(replace(ep.provider_name, " " => "-"))
    provider_normalized in EXCLUDED_PROVIDERS && return true
    startswith(provider_normalized, "echo_") && return true
    return (provider_normalized, lowercase(model_id)) in EXCLUDED_ENDPOINTS
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

        # Filter excluded providers, then dedupe by provider_name (keep first)
        filtered_endpoints = unique(ep -> replace(ep.provider_name, " " => "-"),
                                    filter(ep -> !should_exclude_endpoint(ep, m.id), all_endpoints))
        excluded_count = length(all_endpoints) - length(filtered_endpoints)

        if excluded_count > 0
            println("  Filtered out $excluded_count endpoint(s) (excluded providers or duplicates)")
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

# ---------- Ollama catalog price matching ----------
#
# Ollama Cloud publishes NO per-token pricing (it's a flat-rate subscription /
# GPU-time model). But its models are the same open weights that OpenRouter
# lists with real per-token prices. So we fuzzy-match each Ollama model to its
# OpenRouter catalog twin and inherit that twin's pricing + context length.
# Models with no catalog twin are DROPPED (we'd otherwise ship a $0/free model).

_norm(s) = lowercase(replace(String(s), r"[^a-z0-9]"i => ""))
function _param_size(s)
    m = match(r"(\d+)b", lowercase(String(s)))
    m === nothing ? nothing : parse(Int, m.captures[1])
end
# Leading alphabetic run of a normalized id, e.g. "ministral3" -> "ministral",
# "gptoss20b" -> "gptoss". Groups variants that differ only by version/size.
function _alpha_family(s)
    m = match(r"^[a-z]+", _norm(s))
    m === nothing ? "" : m.match
end

"""
Find the best OpenRouter catalog id for a bare Ollama model id, or `nothing`.

E.g. "gpt-oss:20b" -> "openai/gpt-oss-120b" family, picking the variant that
best matches the version/size. `:free` catalog variants are skipped (they're
\$0 and would defeat the purpose). Scoring prefers an exact normalized match,
then same parameter size, then the tightest family name.

A candidate is considered if either (a) one normalized id is a prefix/substring
of the other, or (b) it shares the alphabetic family AND the exact parameter
size. Path (b) is essential for ids like "ministral-3:14b", where Ollama's "3"
is a generation tag and "14b" the size, so the catalog twin
"mistralai/ministral-14b-2512" shares no substring with base "ministral3".
"""
function find_catalog_match(ollama_id::AbstractString, catalog_ids)
    o_after = lowercase(ollama_id)
    o_norm  = _norm(o_after)
    o_base  = _norm(replace(o_after, r":.*" => ""))   # drop ":size" tag
    o_size  = _param_size(o_after)
    o_fam   = _alpha_family(o_base)
    isempty(o_base) && return nothing

    best, best_score = nothing, -Inf
    for cid in catalog_ids
        after = lowercase(last(split(cid, "/")))
        endswith(after, ":free") && continue
        c_norm = _norm(after)
        c_size = _param_size(after)

        name_match = startswith(c_norm, o_base) || startswith(o_base, c_norm) || occursin(o_base, c_norm)
        size_match = o_size !== nothing && c_size !== nothing && o_size == c_size &&
                     !isempty(o_fam) && _alpha_family(after) == o_fam
        (name_match || size_match) || continue

        score = 0.0
        o_norm == c_norm && (score += 100)
        name_match       && (score += 10)
        if o_size !== nothing && c_size !== nothing
            score += (o_size == c_size ? 50 : -abs(o_size - c_size) / 100)
        end
        score += 1.0 / length(c_norm)

        if score > best_score
            best, best_score = cid, score
        end
    end
    return best
end

"""
Build model specs for the `ollama_cloud` provider from its native `/api/tags`
listing, inheriting pricing + context length from the matching OpenRouter
catalog model (`catalog_specs`). Unmatched models are dropped and reported.
Returns `[]` if the provider is unreachable (missing OLLAMA_API_KEY).
"""
function build_ollama_specs(provider_slug::AbstractString, catalog_specs::Vector)
    local raw
    try
        raw = list_native_models(provider_slug)
    catch err
        @warn "Skipping Ollama models; provider unreachable" provider=provider_slug exception=err
        return Any[]
    end
    println("Fetched $(length(raw)) model(s) from $provider_slug")

    catalog_by_id = Dict(d["id"] => d for d in catalog_specs)
    catalog_ids = collect(keys(catalog_by_id))

    specs = Any[]
    dropped = String[]
    for m in raw
        model_id = get(m, "model", get(m, "name", nothing))
        model_id === nothing && continue

        match_id = find_catalog_match(model_id, catalog_ids)
        if match_id === nothing
            push!(dropped, model_id)   # no catalog twin -> no pricing -> drop
            continue
        end
        twin_ep = catalog_by_id[match_id]["endpoints"][1]

        # Convert ISO `modified_at` (e.g. "2025-12-02T00:00:00Z") to a unix
        # timestamp for the `created` field; take the leading "yyyy-mm-ddTHH:MM:SS".
        created = nothing
        ts = get(m, "modified_at", nothing)
        if ts !== nothing && length(ts) >= 19
            try
                created = round(Int, datetime2unix(DateTime(ts[1:19], dateformat"yyyy-mm-ddTHH:MM:SS")))
            catch
            end
        end

        endpoint = Dict(
            "provider_name" => provider_slug,
            "endpoint_name" => model_id,
            "context_length" => twin_ep["context_length"],
            "max_completion_tokens" => twin_ep["max_completion_tokens"],
            "pricing" => twin_ep["pricing"],          # inherited per-token pricing
            "tag" => "$provider_slug/$model_id",
        )
        # `id` is BARE (no provider prefix); the frontend builds the final slug as
        # `${provider_name}:${id}` from the selected endpoint, matching OpenRouter
        # catalog entries. A prefixed id here would double-prefix to
        # `ollama_cloud:ollama_cloud:...`.
        push!(specs, Dict(
            "id" => model_id,
            "name" => model_id,
            "created" => created,
            "endpoints" => Any[endpoint],
        ))
    end

    if !isempty(dropped)
        println("\n⚠️  Dropping $(length(dropped)) $provider_slug model(s) with no catalog price match:")
        foreach(d -> println("  - $d"), dropped)
    end
    println("Kept $(length(specs)) $provider_slug model(s) with inherited pricing")
    return specs
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

    # Drop models with no reachable endpoints (e.g. ~latest aliases, fully
    # excluded-provider models) — they can't be routed, so don't ship them.
    dropped = filter(d -> isempty(d["endpoints"]), specs)
    if !isempty(dropped)
        println("\nDropping $(length(dropped)) model(s) with no endpoints:")
        foreach(d -> println("  - $(d["id"])"), dropped)
    end
    filter!(d -> !isempty(d["endpoints"]), specs)

    println("\nTotal excluded endpoints: $excluded_endpoints_count")

    # Append native Ollama Cloud models (not in the OpenRouter catalog), with
    # pricing inherited from their matching catalog twins.
    append!(specs, build_ollama_specs("ollama_cloud", specs))

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

    # Write atomically: a temp file in the same dir + rename, so file watchers
    # (e.g. Turbopack) never observe a half-written/empty file mid-export.
    tmp_file = out_file * ".tmp"
    open(tmp_file, "w") do io
        JSON3.pretty(io, data)
    end
    mv(tmp_file, out_file; force=true)

    total_models = data["total_models"]
    println("✅ Exported $total_models models to $out_file")
end

# Run only when executed as a script (tests `include` this file for the matcher).
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end