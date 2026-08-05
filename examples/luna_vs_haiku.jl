# Luna vs Haiku: latency + basic IQ. Run:
#   julia --project=. examples/luna_vs_haiku.jl
using OpenRouter
using OpenRouterCLIProxyAPI: setup_cli_proxy!
using Printf: @printf, @sprintf
using Statistics: mean

setup_cli_proxy!(mutate=true)

const A = "openai:openai/gpt-5.6-luna"
const B = "anthropic:anthropic/claude-haiku-4.5"

const CASES = [
    ("A bat and a ball cost \$1.10. The bat costs \$1.00 more than the ball. How many cents is the ball? Number only.",
        a -> occursin("5", a) && !occursin("10", a)),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how many minutes for 100 machines to make 100 widgets? Number only.",
        a -> occursin("5", a) && !occursin("100", a) && !occursin("500", a)),
    ("A farmer has 17 sheep. All but 9 die. How many are left? Number only.",
        a -> occursin("9", a)),
    ("Continue the sequence: 2, 6, 12, 20, 30, ? Number only.",
        a -> occursin("42", a)),
    ("What is the next prime after 13? Number only.",
        a -> occursin("17", a)),
]

ask(model, prompt) = begin
    t0 = time(); ttft = Ref(NaN); total = Ref(NaN)
    cb = HttpStreamHooks(
        on_start   = () -> (isnan(ttft[]) && (ttft[] = time() - t0); ""),
        on_meta_ai = (tk, c, el) -> (total[] = el === nothing ? time()-t0 : el; ""),
        content_formatter = _ -> "")
    r = aigen(prompt, model; streamcallback=cb, cache=nothing)
    (; text=strip(replace(r.content, r"\s+" => " ")), ttft=ttft[], total=total[],
       out=r.tokens.completion_tokens, cost=r.cost)
end

score = Dict(A => 0, B => 0)
stats = Dict(A => NamedTuple[], B => NamedTuple[])
println(@sprintf("%-56s  %-4s  %-4s", "question", "LUNA", "HAI"))
for (q, grade) in CASES
    ra = ask(A, q); rb = ask(B, q)
    ok_a = grade(ra.text); ok_b = grade(rb.text)
    ok_a && (score[A] += 1); ok_b && (score[B] += 1)
    push!(stats[A], ra); push!(stats[B], rb)
    @printf("%-56s   %s    %s\n", first(q, 54), ok_a ? "✅" : "❌", ok_b ? "✅" : "❌")
end

report(name, model) = begin
    s = stats[model]
    @printf("%-18s  IQ %d/%-2d  ttft %.2fs  total %.2fs  out %.0f tok  \$%.5f\n",
        name, score[model], length(CASES),
        mean(x->x.ttft, s), mean(x->x.total, s), mean(x->x.out, s), sum(x->x.cost, s))
end
println("\n════ METRICS (avg over $(length(CASES)) questions) ════")
report("Luna (gpt-5.6)", A)
report("Claude Haiku-4.5", B)
