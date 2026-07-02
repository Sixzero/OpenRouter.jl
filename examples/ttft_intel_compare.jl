# TTFT + intelligence probe across frontier models (Anthropic/OpenAI proxy + z.ai + Cerebras).
# Priorities: TTFT first, then reasoning correctness. Cost/throughput secondary.
#
# "think" column = completion tokens spent before the answer (reasoning). High = model is
# thinking a lot (inflates latency); low/0 = direct. This surfaces who's reasoning by default.
#
# Run: cd todoforai/agent && julia --project=. ../../OpenRouter.jl/examples/ttft_intel_compare.jl

using OpenRouter
using Statistics: median
using Printf: @printf
using OpenRouterCLIProxyAPI: setup_cli_proxy!
let f = joinpath(@__DIR__, "..", "..", "todoforai", "agent", "scripts", "load_env.jl")
    isfile(f) && (include(f); Base.invokelatest(load_env))
end
setup_cli_proxy!(mutate=true)
# Gemini (google-ai-studio) needs a direct key; the working one lives in GOOGLE_API_KEY_2.
haskey(ENV, "GOOGLE_API_KEY_2") && (ENV["GOOGLE_API_KEY"] = ENV["GOOGLE_API_KEY_2"])

const MODELS = haskey(ENV, "MODELS") ? split(ENV["MODELS"], ",") : [
    "anthropic:anthropic/claude-haiku-4.5",
    "anthropic:anthropic/claude-sonnet-5",
    "anthropic:anthropic/claude-opus-4.5",
    "openai:openai/gpt-5.4",
    "openai:openai/gpt-5.4-mini",
    "z.ai:z-ai/glm-5.2",
    "cerebras:z-ai/glm-4.7",         # GLM-4.7 on Cerebras hardware — knowledgeable + very fast
    "cerebras:openai/gpt-oss-120b",  # open-weight 120B on Cerebras
    "google-ai-studio:google/gemini-2.5-flash",       # fast + accurate
    "google-ai-studio:google/gemini-2.5-flash-lite",  # fastest Gemini
]
const REPS = parse(Int, get(ENV, "REPS", "3"))

const TTFT_PROMPT = "Reply with just the word: ok"

# Harder intelligence probe than bat-and-ball (which everything passes). Multi-step logic.
const IQ_PROMPT = """Alice is twice as old as Bob was when Alice was as old as Bob is now.
Bob is 30. How old is Alice? Reply with ONLY the number."""
const IQ_ANSWER = "40"

measure(model, prompt) = begin
    t0 = time(); ttft = Ref(NaN); total = Ref(NaN)
    content = IOBuffer()
    cb = HttpStreamHooks(
        on_start   = () -> (isnan(ttft[]) && (ttft[] = time() - t0); ""),
        on_meta_ai = (tokens, cost, elapsed) -> (total[] = elapsed === nothing ? time() - t0 : elapsed; ""),
        content_formatter = t -> (print(content, t); ""),
    )
    r = aigen(prompt, model; streamcallback=cb, cache=nothing)
    isnan(total[]) && (total[] = time() - t0)
    isnan(ttft[]) && (ttft[] = total[])   # non-streaming providers: use total as TTFT
    reasoning = (hasproperty(r, :reasoning) && r.reasoning !== nothing) ? length(split(r.reasoning)) : 0
    (; ttft=ttft[], total=total[], text=strip(String(take!(content))), reasoning, cost=r.cost)
end

agg = Dict{String,NamedTuple}()
for m in MODELS
    m = String(strip(m)); println("═══ $m ═══")
    try; measure(m, TTFT_PROMPT); catch e; @warn "skip $m" exception=e; continue; end  # warmup
    ttfts = Float64[]
    for _ in 1:REPS; push!(ttfts, measure(m, TTFT_PROMPT).ttft); end
    iq = measure(m, IQ_PROMPT)
    correct = occursin(IQ_ANSWER, iq.text)
    agg[m] = (; ttft=median(ttfts), correct, ans=first(iq.text, 14), think=iq.reasoning, cost=iq.cost)
    a = agg[m]
    @printf("  ttft %5.2fs  IQ %s (\"%s\")  think~%d words\n", a.ttft, a.correct ? "✅" : "❌", a.ans, a.think)
end

println("\n════════ TTFT × Intelligence (sorted by TTFT) ════════")
@printf("%-40s %8s %5s %8s   %s\n", "model", "ttft", "IQ", "think", "answer")
for m in sort(collect(keys(agg)), by=k->agg[k].ttft)
    a = agg[m]
    @printf("%-40s %7.2fs %5s %8d   %s\n", m, a.ttft, a.correct ? "✅" : "❌", a.think, a.ans)
end
println("\nttft = starts fast · IQ = multi-step logic · think = reasoning words before answering")
