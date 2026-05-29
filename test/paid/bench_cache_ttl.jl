# Cache TTL Benchmark
# Measures how long Anthropic prompt caching stays alive (default TTL is 5 min).
#
# Why not a single seed + incremental probes?
#   Anthropic refreshes the cache TTL on every *read*, not just writes. So probing
#   one shared cache every <5 min keeps it alive forever and never finds the TTL.
#
# Strategy (independent caches on one timeline):
#   1. Seed N caches at t=0, each with a *distinct* system prompt → distinct cache
#      key. Each call with cache=:system → expects cache_write > 0.
#   2. Probe cache i once at its target age. Reading cache i does NOT refresh the
#      others (different keys), so every probe is an independent raw-TTL measurement
#      while sharing the same ~6 min wall clock.
#   3. cache_read > 0 → still alive (HIT); cache_read == 0 → expired (MISS).
#      TTL lies between the largest HIT age and the smallest MISS age.
#
# Measured result (claude-haiku-4.5, 2026-05-29):
#   Default ephemeral TTL ≈ 5 min (300s). A coarse sweep gave HIT @5:01, MISS @5:31;
#   a 5s-resolution sweep narrowed it to HIT @5:07, MISS @5:15 — i.e. expiry ~300s,
#   the few seconds of slack being request round-trip latency on top of the seed time.
#
# Usage:
#   julia --project=. test/paid/bench_cache_ttl.jl
#
# NOTE: Long-running (~6min+), not a unit test. Costs real money (tiny — haiku,
#       ~5.6K cached tokens per slot, N+1 cache writes ≈ a few cents total).
#       Min cacheable prompt for Haiku 4.5 is 4096 tokens; ours is ~7.4K.

using OpenRouter
using OpenRouter: UserMessage, SystemMessage
using Dates

const MODEL = "anthropic:anthropic/claude-haiku-4.5"
const MAX_TOKENS = 50

# Target cache ages (seconds from each slot's seed): 3:00, 4:00, 4:30, 5:00, 5:30, 6:00
const TARGETS = [180, 240, 270, 300, 330, 360]

# Unique per run so re-running within the TTL window can't inherit a still-warm cache
# from a previous run (which would corrupt slot ages → always writes fresh caches).
const RUN_ID = Dates.format(now(), "yyyymmddHHMMSS")

# Distinct system prompt per slot → distinct cache key. Unique marker first so the
# cached prefix differs from the first token. Body keeps it well above the 4096-token
# Haiku 4.5 minimum for caching.
build_system_prompt(slot) = "[run $RUN_ID slot #$slot] You are a helpful assistant. " *
    ("Here is a long repeated sentence to ensure the system prompt exceeds the minimum token threshold required for Anthropic prompt caching to activate. "^200)

function make_call(slot, user_msg::String; cache_mode=:system)
    r = aigen([
        SystemMessage(content=build_system_prompt(slot)),
        UserMessage(content=user_msg),
    ], MODEL; cache=cache_mode, max_tokens=MAX_TOKENS)
    return (
        read = r.tokens.input_cache_read,
        write = r.tokens.input_cache_write,
        cost = something(r.cost, 0.0),
    )
end

fmt_age(s) = (t = round(Int, s); "$(t ÷ 60):$(lpad(t % 60, 2, '0'))")

function main()
    println("=" ^ 60)
    println("Cache TTL Benchmark — $(now())")
    println("Model: $MODEL")
    println("Slots: $(length(TARGETS)) | Targets: $(join(fmt_age.(TARGETS), ", "))")
    println("=" ^ 60)

    total_cost = 0.0

    # --- Sanity: caching mechanics work (separate slot, never timed) ---
    println("\n[sanity] Seeding slot 0...")
    s_seed = make_call(0, "Say hello."; cache_mode=:system)
    total_cost += s_seed.cost
    println("    write=$(s_seed.write) read=$(s_seed.read) cost=$(round(s_seed.cost; digits=6))")
    @assert s_seed.write > 0 "Expected cache_write > 0 when seeding"

    s_probe = make_call(0, "Say goodbye.")
    total_cost += s_probe.cost
    println("    write=$(s_probe.write) read=$(s_probe.read) cost=$(round(s_probe.cost; digits=6))")
    @assert s_probe.read > 0 "Expected cache_read > 0 on immediate follow-up"

    # --- Seed all timed caches at ~t=0 ---
    println("\nSeeding $(length(TARGETS)) timed caches...")
    seed_times = Float64[]
    for slot in eachindex(TARGETS)
        r = make_call(slot, "Seed slot $slot."; cache_mode=:system)
        push!(seed_times, time())
        total_cost += r.cost
        @assert r.write > 0 "Expected cache_write > 0 when seeding slot $slot"
        println("    slot $slot: write=$(r.write)")
    end

    # --- Probe each slot at its target age (ascending → sequential sleeps) ---
    results = NamedTuple[]
    for slot in eachindex(TARGETS)
        target = TARGETS[slot]
        due = seed_times[slot] + target
        wait_s = due - time()
        if wait_s > 0
            println("\n    Waiting $(round(Int, wait_s))s (slot $slot → age $(fmt_age(target)))...")
            sleep(wait_s)
        end

        r = make_call(slot, "Probe slot $slot.")
        total_cost += r.cost
        age = time() - seed_times[slot]
        hit = r.read > 0
        push!(results, (slot=slot, target=target, age=age, hit=hit, read=r.read, write=r.write))
        println("[$(fmt_age(target))] slot $slot: read=$(r.read) write=$(r.write) age=$(fmt_age(age)) → $(hit ? "HIT ✓" : "MISS ✗")")
    end

    # --- Summary ---
    println("\n" * "=" ^ 60)
    println("Results (TTL is between the last HIT and first MISS):")
    for r in results
        println("  age $(fmt_age(r.age)) (target $(fmt_age(r.target))): $(r.hit ? "HIT ✓" : "MISS ✗")")
    end

    last_hit = findlast(r -> r.hit, results)
    first_miss = findfirst(r -> !r.hit, results)
    if first_miss === nothing
        println("\n→ Cache still alive at all probes (≥ $(fmt_age(results[end].age))). Extend TARGETS.")
    elseif last_hit === nothing
        println("\n→ Cache expired before the first probe (≤ $(fmt_age(results[1].age))). Shorten TARGETS.")
    else
        println("\n→ TTL between $(fmt_age(results[last_hit].age)) and $(fmt_age(results[first_miss].age))")
    end

    println("\nTotal cost: \$$(round(total_cost; digits=6))")
    println("Benchmark complete — $(now())")
    println("=" ^ 60)
end

main()
