# Head-to-head intelligence: Cerebras GLM-4.7 vs Claude Haiku-4.5.
# Same probes, grade each answer. Run:
#   cd todoforai/agent && julia --project=. ../../OpenRouter.jl/examples/glm47_vs_haiku.jl

using OpenRouter
using Printf: @printf, @sprintf
using OpenRouterCLIProxyAPI: setup_cli_proxy!
let f = joinpath(@__DIR__, "..", "..", "todoforai", "agent", "scripts", "load_env.jl")
    isfile(f) && (include(f); Base.invokelatest(load_env))
end
setup_cli_proxy!(mutate=true)

const A = "cerebras:z-ai/glm-4.7"
const B = "anthropic:anthropic/claude-haiku-4.5"

# (prompt, grader) — grader(answer)::Bool
const CASES = [
    ("A bat and a ball cost \$1.10. The bat costs \$1.00 more than the ball. How many cents is the ball? Number only.",
        a -> occursin("5", a) && !occursin("10", a)),
    ("Alice is twice as old as Bob was when Alice was as old as Bob is now. Bob is 30. How old is Alice? Number only.",
        a -> occursin("40", a)),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how many minutes for 100 machines to make 100 widgets? Number only.",
        a -> occursin("5", a) && !occursin("100", a) && !occursin("500", a)),
    ("A farmer has 17 sheep. All but 9 die. How many are left? Number only.",
        a -> occursin("9", a)),
    ("Which is heavier: a kilogram of feathers or a kilogram of steel? One word.",
        a -> occursin(r"same|equal|neither"i, a)),
    ("I have 3 apples. I eat 1 and give away 1. Then I buy a dozen more. How many apples now? Number only.",
        a -> occursin("13", a)),
    ("Continue the sequence: 2, 6, 12, 20, 30, ? Number only.",
        a -> occursin("42", a)),
    ("A rope reaches from the top of a 20m pole to a point 15m from its base. How long is the rope? Number only (meters).",
        a -> occursin("25", a)),
    ("Sort these by size, smallest first: elephant, ant, dog, whale. List only.",
        a -> (findfirst("ant", lowercase(a)) !== nothing) && findfirst("ant", lowercase(a)).start < findfirst("whale", lowercase(a)).start),
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

using Statistics: mean
score = Dict(A => 0, B => 0)
stats = Dict(A => NamedTuple[], B => NamedTuple[])
println(@sprintf("%-56s  %-4s  %-4s", "question", "GLM", "HAI"))
for (q, grade) in CASES
    ra = ask(A, q); rb = ask(B, q)
    ok_a = grade(ra.text); ok_b = grade(rb.text)
    ok_a && (score[A] += 1); ok_b && (score[B] += 1)
    push!(stats[A], ra); push!(stats[B], rb)
    @printf("%-56s   %s    %s\n", first(q, 54), ok_a ? "✅" : "❌", ok_b ? "✅" : "❌")
    (!ok_a || !ok_b) && println("      GLM: $(first(ra.text,60))\n      HAI: $(first(rb.text,60))")
end

report(name, model) = begin
    s = stats[model]
    @printf("%-18s  IQ %d/%-2d  ttft %.2fs  total %.2fs  out %.0f tok  \$%.5f\n",
        name, score[model], length(CASES),
        mean(x->x.ttft, s), mean(x->x.total, s), mean(x->x.out, s), sum(x->x.cost, s))
end
println("\n════ METRICS (avg over $(length(CASES)) questions) ════")
report("Cerebras GLM-4.7", A)
report("Claude Haiku-4.5", B)
