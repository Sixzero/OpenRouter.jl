using Test
using OpenRouter
using OpenRouter: with_thinking_suffix, get_provider_info, ProviderInfo

@testset "thinking suffix -> reasoning_effort" begin
    meta = get_provider_info("meta")
    @test meta.thinking_suffix == :reasoning_effort
    id, kw = with_thinking_suffix(meta, "muse-wonder-1.0(high)", pairs((; max_tokens=10)))
    @test id == "muse-wonder-1.0"
    @test kw[:reasoning_effort] == "high" && kw[:max_tokens] == 10
    # explicit kwarg wins, suffix still stripped
    id, kw = with_thinking_suffix(meta, "muse-wonder-1.0(high)", pairs((; reasoning_effort="low")))
    @test id == "muse-wonder-1.0" && kw[:reasoning_effort] == "low"
    # numeric budget / no suffix untouched
    @test with_thinking_suffix(meta, "muse-wonder-1.0(16384)", pairs((;)))[1] == "muse-wonder-1.0(16384)"
    @test with_thinking_suffix(meta, "muse-wonder-1.0", pairs((;)))[1] == "muse-wonder-1.0"
    # proxy providers keep the suffix on the id
    anth = get_provider_info("anthropic")
    @test anth.thinking_suffix == :passthrough
    @test with_thinking_suffix(anth, "claude-opus-4.7(high)", pairs((;)))[1] == "claude-opus-4.7(high)"
end
