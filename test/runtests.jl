using OpenRouter
using OpenRouter: update_db, calculate_cost, ChatCompletionSchema
using Test
using Aqua

@testset "OpenRouter.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(OpenRouter)
    end
    
    @testset "Basic functionality" begin
        @test isa(OpenRouter.list_models, Function)
    end
    
    @testset "Cache operations" begin
        if isempty(get(ENV, "OPENROUTER_API_KEY", ""))
            @info "Skipping cache operations tests: OPENROUTER_API_KEY not set"
        else
            cache = update_db(fetch_endpoints = false)
            @test length(cache.models) > 0

            # Pick one model id
            any_id = first(keys(cache.models))
            cm = get_model(any_id, fetch_endpoints = false)
            @test cm !== nothing
            @test cm.model.id == any_id
        end
    end
    
    @testset "Provider model parsing" begin
        provider_info, model_id, endpoint = OpenRouter.parse_provider_model("Together:moonshotai/kimi-k2-thinking")
        @test provider_info.base_url == "https://api.together.xyz/v1"
        @test model_id == "moonshotai/kimi-k2-thinking"
        @test endpoint isa OpenRouter.ProviderEndpoint
    end

    @testset "cost calculation with cache" begin
        pricing = Pricing(
            prompt = "0.000002",
            completion = "0.000004",
            request = "0",
            image = "0",
            web_search = "0",
            internal_reasoning = nothing,
            image_output = nothing,
            audio = nothing,
            input_audio_cache = nothing,
            input_cache_read = "0.000001",
            input_cache_write = "0.0000015",
            discount = nothing
        )
        tokens = Dict(
            :prompt_tokens => 1000,
            :completion_tokens => 500,
            :input_cache_read => 200,
            :input_cache_write => 100,
        )
        cost = calculate_cost(pricing, tokens)
        @test isapprox(cost, 1000*0.000002 + 500*0.000004 + 200*0.000001 + 100*0.0000015; atol=1e-10)
    end

    @testset "calculate_cost with ProviderEndpoint" begin
        pricing = Pricing(
            prompt = "0.000002",
            completion = "0.000004",
            request = "0",
            image = "0",
            web_search = "0",
            internal_reasoning = nothing,
            image_output = nothing,
            audio = nothing,
            input_audio_cache = nothing,
            input_cache_read = "0.000001",
            input_cache_write = "0.0000015",
            discount = nothing,
        )
        endpoint = ProviderEndpoint(
            name = "test-endpoint",
            model_name = "test-model",
            context_length = 8192,
            pricing = pricing,
            provider_name = "test-provider",
            tag = nothing,
            quantization = nothing,
            max_completion_tokens = nothing,
            max_prompt_tokens = nothing,
            supported_parameters = nothing,
            uptime_last_30m = nothing,
            supports_implicit_caching = nothing,
            status = nothing
        )
        tokens = Dict(
            :prompt_tokens => 1000,
            :completion_tokens => 500,
            :input_cache_read => 200,
            :input_cache_write => 100,
        )
        cost = calculate_cost(endpoint, tokens)
        @test isapprox(cost, 1000*0.000002 + 500*0.000004 + 200*0.000001 + 100*0.0000015; atol=1e-10)
    end

    @testset "ChatCompletion stream payload" begin
        # When stream=true, opt into usage stats so providers like Moonshot's
        # kimi-k2.6 emit a trailing usage chunk during streaming.
        p_stream = OpenRouter.build_payload(ChatCompletionSchema(), "hi", "any-model", nothing, true)
        @test p_stream["stream"] === true
        @test p_stream["stream_options"] == Dict("include_usage" => true)

        # When stream=false, no stream_options are added.
        p_nostream = OpenRouter.build_payload(ChatCompletionSchema(), "hi", "any-model", nothing, false)
        @test !haskey(p_nostream, "stream")
        @test !haskey(p_nostream, "stream_options")
    end

    @testset "ChatCompletion is_done semantics" begin
        using OpenRouter: is_done, StreamChunk
        import JSON3
        sch = ChatCompletionSchema()

        mkchunk(d) = StreamChunk(data=JSON3.write(d), json=JSON3.read(JSON3.write(d)))

        # finish_reason alone (no usage) → not done; trailing usage chunk still expected.
        @test !is_done(sch, mkchunk(Dict("choices" => [Dict("finish_reason" => "stop", "delta" => Dict())])))

        # Trailing usage-only chunk (empty choices, has usage) → done.
        @test is_done(sch, mkchunk(Dict("choices" => [], "usage" => Dict("prompt_tokens" => 5, "completion_tokens" => 3, "total_tokens" => 8))))

        # usage + finish_reason on same chunk (DeepSeek style) → done.
        @test is_done(sch, mkchunk(Dict("choices" => [Dict("finish_reason" => "stop", "delta" => Dict())], "usage" => Dict("prompt_tokens" => 5, "completion_tokens" => 3, "total_tokens" => 8))))

        # [DONE] sentinel → done.
        @test is_done(sch, StreamChunk(data="[DONE]", json=nothing))

        # Mid-stream content chunk → not done.
        @test !is_done(sch, mkchunk(Dict("choices" => [Dict("delta" => Dict("content" => "x"))])))
    end

    @testset "Anthropic drops deprecated sampling params" begin
        using OpenRouter: AnthropicSchema, build_payload
        drops(m) = let p = build_payload(AnthropicSchema(), "hi", m, nothing; top_p=0.9, temperature=0.7, max_tokens=10)
            !haskey(p, "top_p") && !haskey(p, "temperature")
        end
        # Opus >= 4.7, Sonnet >= 5 and Fable deprecated `top_p`/`temperature` (API 400 otherwise).
        @test drops("anthropic/claude-opus-4.7")
        @test drops("anthropic/claude-opus-4.8")
        @test drops("anthropic/claude-opus-5")
        @test drops("anthropic/claude-sonnet-5")
        @test drops("anthropic/claude-fable-5")
        # Thinking suffix also forces a drop on any model.
        @test drops("anthropic/claude-sonnet-4.6(xhigh)")
        # Older models keep them.
        @test !drops("anthropic/claude-opus-4.6")
        @test !drops("anthropic/claude-sonnet-4.6")
    end

    @testset "Streaming error without Content-Type" begin
        using OpenRouter: HttpStreamCallback, HttpStreamHooks, ChatCompletionSchema, streamed_request!
        import HTTP

        # Mimic z.ai: error status with NO Content-Type header. Must surface the real
        # API message, not the "Content-Type header must be present and unique" assertion.
        handler(_) = HTTP.Response(429, String[]; body="""{"error":{"message":"Insufficient balance"}}""")
        server = HTTP.serve!(handler, "127.0.0.1", 18799)
        try
            url = "http://127.0.0.1:18799/v1/chat/completions"
            for cb in (HttpStreamCallback(; schema=ChatCompletionSchema()),
                       HttpStreamHooks(; schema=ChatCompletionSchema()))
                err = try
                    streamed_request!(cb, url, ["Content-Type" => "application/json"], "{}")
                    nothing
                catch e
                    e
                end
                @test err isa HTTP.RequestError
                @test occursin("429", err.error)
                @test occursin("Insufficient balance", err.error)
            end
        finally
            close(server)
        end
    end

    # Include custom provider tests
    include("test_custom_providers.jl")

    # Tool message serialization tests
    include("test_tool_messages.jl")

    # Ollama native API tests (live cloud tests gated on OLLAMA_API_KEY)
    include("test_ollama_provider.jl")

    # Ollama -> OpenRouter catalog price matching (export script)
    include("test_ollama_pricing_match.jl")

    # Echo server tests (all schemas)
    include("test_echo_server.jl")
end
