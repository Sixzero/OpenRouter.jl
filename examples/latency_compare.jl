# Multi-model latency + quality comparison — based on stream_hooks_example.jl.
# Per model, over the same prompts (× reps), measures:
#   ttft    — time to first streamed token (on_start)
#   total   — full generation wall time (elapsed)
#   tok/s   — output throughput (completion_tokens / total)
#   in/out  — prompt/completion tokens (out ≈ how much it said to answer)
#   cost    — USD
# A warmup pass per model absorbs cold-start/precompile so numbers are steady-state.
# Caching disabled so in_tok/latency aren't skewed by warm prefixes.
#
# Run:  cd todoforai/agent && julia --project=. ../../OpenRouter.jl/examples/latency_compare.jl
# Tune: MODELS=..., REPS=3 julia ...

using OpenRouter
using Statistics: median
using Printf: @printf

if get(ENV, "USE_CLI_PROXY", "1") == "1"
    using OpenRouterCLIProxyAPI: setup_cli_proxy!
    let f = joinpath(dirname(@__DIR__), "..", "todoforai", "agent", "scripts", "load_env.jl")
        isfile(f) && (include(f); Base.invokelatest(load_env))
    end
    setup_cli_proxy!(mutate=true)
end

const MODELS = haskey(ENV, "MODELS") ? split(ENV["MODELS"], ",") : [
    "anthropic:anthropic/claude-sonnet-4.5",
    "anthropic:anthropic/claude-haiku-4.5",
    "anthropic:anthropic/claude-sonnet-5",
    "openai:openai/gpt-5.4-mini",
    "openai:openai/gpt-5.5",
]

const REPS = parse(Int, get(ENV, "REPS", "3"))

# Short + longer prompt to separate startup cost from streaming throughput.
const PROMPTS = [
    "Count to 1-5 in 1 line:",
    "In 2 sentences, explain when to use a CLI over a GUI.",
]

run_one(model, prompt) = begin
    t0 = time()
    ttft = Ref(NaN); total = Ref(NaN)
    cb = HttpStreamHooks(
        on_start   = () -> (isnan(ttft[]) && (ttft[] = time() - t0); ""),
        on_meta_ai = (tokens, cost, elapsed) -> (total[] = elapsed === nothing ? time() - t0 : elapsed; ""),
        content_formatter = _ -> "",
    )
    r = aigen(prompt, model; streamcallback=cb, cache=nothing)
    (; ttft=ttft[], total=total[], intok=r.tokens.prompt_tokens, outtok=r.tokens.completion_tokens, cost=r.cost)
end

agg = Dict{String,NamedTuple}()
for m in MODELS
    m = String(strip(m))
    println("═══ $m ═══")
    try
        run_one(m, PROMPTS[1])  # warmup (discarded)
    catch e
        @warn "warmup failed for $m" exception=e; continue
    end
    samples = NamedTuple[]
    for p in PROMPTS, _ in 1:REPS
        s = run_one(m, p)
        push!(samples, s)
    end
    ttfts  = [s.ttft for s in samples]
    totals = [s.total for s in samples]
    outs   = [s.outtok for s in samples]
    toks   = [s.outtok / max(s.total, 1e-6) for s in samples]
    agg[m] = (; ttft=median(ttfts), total=median(totals), toks=median(toks),
                intok=median([s.intok for s in samples]), out=median(outs),
                cost=sum(s.cost for s in samples))
    a = agg[m]
    @printf("  ttft %5.2fs  total %5.2fs  %5.1f tok/s  out %3.0f  \$%.6f (%d runs)\n",
        a.ttft, a.total, a.toks, a.out, a.cost, length(samples))
end

println("\n════════ Comparison (median over prompts × $REPS reps) ════════")
@printf("%-42s %8s %8s %9s %7s %7s %10s\n", "model", "ttft", "total", "tok/s", "in", "out", "cost_sum")
for m in MODELS
    m = String(strip(m)); haskey(agg, m) || continue
    a = agg[m]
    @printf("%-42s %7.2fs %7.2fs %8.1f %7.0f %7.0f %10.6f\n",
        m, a.ttft, a.total, a.toks, a.intok, a.out, a.cost)
end
println("\nttft = starts fast · total = whole gen · tok/s = stream speed · out = verbosity to answer")
