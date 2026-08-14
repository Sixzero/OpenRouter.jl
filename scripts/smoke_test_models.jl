#!/usr/bin/env julia
# Live smoke-test: for each native provider we hold an API key for, call its
# NEWEST catalog model (from the frontend export) with a tiny prompt and report
# whether it actually works. Answers "which models do we even support?".
#
# Usage:
#   julia --project=. scripts/smoke_test_models.jl [path/to/models_data.json]
#
# Keys are read from the agent .env.production (KEY=value lines) into ENV.

using OpenRouter
using JSON3

const ENV_FILE = get(ENV, "SMOKE_ENV_FILE",
    joinpath(homedir(), "repo/todoforai/agent/.env.production"))
const MODELS_JSON = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(homedir(), "repo/todoforai/frontend/src/assets/models_data.json")

# ---- load env keys ----
function load_env!(path)
    isfile(path) || (@warn "env file missing" path; return)
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#") || !occursin("=", s)) && continue
        k, v = split(s, "=", limit=2)
        k = strip(k); v = strip(v)
        (startswith(v, "\"") && endswith(v, "\"")) && (v = v[2:end-1])
        # keep first occurrence (primary key), don't clobber
        haskey(ENV, k) || (ENV[k] = v)
    end
end

# normalize like get_provider_info (drop -._, lowercase)
norm(s) = replace(lowercase(String(s)), r"[-._]" => "")

load_env!(ENV_FILE)

# Refresh the local model/endpoint cache so provider-host lookups reflect live
# OpenRouter routing (a stale cache causes false "provider does not host" fails).
if get(ENV, "SMOKE_SKIP_REFRESH", "") == ""
    try
        @info "Refreshing model DB (set SMOKE_SKIP_REFRESH=1 to skip)…"
        update_db(full_refresh=true)
    catch e
        # A failed refresh means endpoint metadata may be stale/missing, which can
        # masquerade as "provider does not host model". Abort rather than report
        # misleading failures — the whole point of this tool is trustworthy results.
        @error "update_db failed; results would be unreliable. " *
               "Set SMOKE_SKIP_REFRESH=1 to run against the existing cache anyway." *
               " err=$(first(sprint(showerror, e), 200))"
        exit(1)
    end
end

# ---- pick newest model per provider from export ----
data = JSON3.read(read(MODELS_JSON, String))
# provider_name (normalized) -> (created, model_id, raw_provider_name)
newest = Dict{String,Tuple{Int,String,String}}()
for m in data.models
    created = m.created === nothing ? 0 : Int(m.created)
    for ep in m.endpoints
        pn = String(ep.provider_name)
        key = norm(pn)
        cur = get(newest, key, (-1, "", ""))
        created > cur[1] && (newest[key] = (created, String(m.id), pn))
    end
end

# providers we can actually route natively (in PROVIDER_INFO) AND hold a key for
function keyed_slug(pn_norm)
    for slug in OpenRouter.list_known_providers()
        norm(slug) == pn_norm || continue
        info = OpenRouter.get_provider_info(slug)
        info === nothing && return nothing
        ev = info.api_key_env_var
        ev === nothing && return slug           # keyless (ollama/echo)
        (haskey(ENV, ev) && !isempty(ENV[ev])) && return slug
        return nothing                          # known but no key
    end
    return nothing
end

targets = Tuple{String,String,String}[]  # (slug, model_id, provider_name)
for (pnn, (_, mid, pn)) in newest
    slug = keyed_slug(pnn)
    slug === nothing && continue
    startswith(norm(slug), "echo") && continue
    push!(targets, (slug, mid, pn))
end
sort!(targets, by = t -> t[1])

# Per-call wall-clock cap so a provider that accepts but never answers can't stall
# the whole run (aigen exposes no native read timeout).
const CALL_TIMEOUT_S = parse(Float64, get(ENV, "SMOKE_TIMEOUT", "60"))

println("Testing $(length(targets)) provider(s) with a key:\n")
results = NamedTuple[]
for (slug, mid, pn) in targets
    spec = "$slug:$mid"
    print(rpad(spec, 55))
    t0 = time()
    ok = false; msg = ""
    t = @async try
        r = aigen("Reply with exactly: OK", spec)
        (true, replace(first(String(r.content), 40), "\n" => " "))
    catch e
        (false, first(sprint(showerror, e), 120))
    end
    timedout = timedwait(() -> istaskdone(t), CALL_TIMEOUT_S) === :timed_out
    if timedout
        ok = false; msg = "TIMEOUT after $(CALL_TIMEOUT_S)s"
    else
        ok, msg = fetch(t)
    end
    dt = round(time() - t0, digits=1)
    println(ok ? "✅ $(dt)s  \"$msg\"" : "❌ $(dt)s  $msg")
    push!(results, (; slug, model=mid, ok, dt, msg))
end

println("\n=== summary ===")
okc = count(r -> r.ok, results)
println("$okc/$(length(results)) OK")
for r in filter(r -> !r.ok, results)
    println("  ❌ $(r.slug):$(r.model) — $(r.msg)")
end
