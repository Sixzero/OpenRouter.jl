using OpenRouter
using OpenRouter: AIMessage, extract_reasoning, extract_tokens
using OpenRouterCLIProxyAPI
using Test

# Requires CLIProxyAPI running locally with valid Anthropic auth.
setup_cli_proxy!(; mutate=true)

@testset "Anthropic thinking via proxy suffix" begin
    PROMPT = "Prove there are infinitely many primes. Think carefully."

    @testset "sonnet-4.5(xhigh) returns thinking block (non-streaming)" begin
        raw = aigen_raw(PROMPT, "anthropic:anthropic/claude-sonnet-4.5(xhigh)"; max_tokens=16000)
        types = [get(b, "type", nothing) for b in get(raw.result, "content", [])]
        @test "thinking" in types
        @test "text" in types

        msg = AIMessage(raw.schema, raw.result; endpoint=raw.provider_endpoint)
        @test msg.reasoning !== nothing && !isempty(msg.reasoning)
        # Anthropic does not expose reasoning tokens in usage -> 0 expected.
        @test msg.tokens.internal_reasoning == 0
    end

    @testset "sonnet-4.5(xhigh) streaming preserves thinking block" begin
        cb = HttpStreamHooks(out=devnull)
        msg = aigen(PROMPT, "anthropic:anthropic/claude-sonnet-4.5(xhigh)";
                    streamcallback=cb, max_tokens=16000)
        # Streaming must reconstruct the thinking content block (regression test).
        @test msg.reasoning !== nothing
        @test msg.tokens.internal_reasoning == 0
    end

    @testset "opus-4.7(xhigh) returns thinking summary" begin
        # Requires the `interleaved-thinking-2025-05-14` beta header (set by
        # override_providers! in OpenRouterCLIProxyAPI); without it opus models
        # return thinking blocks with empty text.
        # Opus thinking is adaptive: use a compute-style prompt that reliably
        # triggers it (the primes proof is sometimes answered without thinking).
        COMPUTE_PROMPT = "How many primes between 300 and 360? Verify each candidate carefully."
        raw = aigen_raw(COMPUTE_PROMPT, "anthropic:anthropic/claude-opus-4.7(xhigh)"; max_tokens=16000)
        types = [get(b, "type", nothing) for b in get(raw.result, "content", [])]
        @test "thinking" in types

        msg = AIMessage(raw.schema, raw.result; endpoint=raw.provider_endpoint)
        @test msg.reasoning !== nothing && !isempty(msg.reasoning)
        @test msg.tokens.internal_reasoning == 0
    end

    @testset "opus-4.6(xhigh) — proxy rejects level" begin
        # opus-4.6 supports adaptive thinking levels low/medium/high/max but NOT xhigh.
        @test_throws Exception aigen_raw(PROMPT,
            "anthropic:anthropic/claude-opus-4.6(xhigh)"; max_tokens=16000)
    end
end
