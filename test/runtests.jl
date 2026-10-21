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
        provider_info, model_id, endpoint = OpenRouter.parse_provider_model("Together:meta-llama/llama-3.3-70b-instruct")
        @test provider_info.base_url == "https://api.together.xyz/v1"
        @test model_id == "meta-llama/llama-3.3-70b-instruct"
        @test endpoint isa OpenRouter.ProviderEndpoint
    end

    @testset "vertex: slug = Gemini API, GeminiSchema" begin
        provider_info, model_id, _ = OpenRouter.parse_provider_model("vertex:google/gemini-4-argon")
        @test provider_info.base_url == "https://generativelanguage.googleapis.com/v1beta"
        @test provider_info.api_key_env_var == "GOOGLE_API_KEY"
        @test provider_info.schema isa OpenRouter.GeminiSchema
        @test model_id == "gemini-4-argon"
    end

    @testset "openai_eu: EU host, openai pricing +10%" begin
        info, model_id, ep = OpenRouter.parse_provider_model("openai_eu:openai/gpt-6.1-sol")
        _, _, base = OpenRouter.parse_provider_model("openai:openai/gpt-6.1-sol")
        @test info.base_url == "https://eu.api.openai.com/v1"
        @test info.api_key_env_var == "OPENAI_API_KEY"
        @test info.schema isa OpenRouter.ResponseSchema
        @test model_id == "gpt-6.1-sol"
        @test parse(Float64, ep.pricing.prompt) ≈ 1.1 * parse(Float64, base.pricing.prompt)
        @test parse(Float64, ep.pricing.completion) ≈ 1.1 * parse(Float64, base.pricing.completion)
    end

    @testset "azure: v1 Responses API on our resource, openai pricing +10%" begin
        withenv("AZURE_OPENAI_HOST" => "tfa-eu") do
            info, model_id, ep = OpenRouter.parse_provider_model("azure:openai/gpt-6.1-sol")
            _, _, base = OpenRouter.parse_provider_model("openai:openai/gpt-6.1-sol")
            @test OpenRouter.resolve_base_url(info.base_url) == "https://tfa-eu.openai.azure.com/openai/v1"
            @test info.schema isa OpenRouter.ResponseSchema
            @test model_id == "gpt-6.1-sol"
            @test ep.provider_name == "azure"
            @test parse(Float64, ep.pricing.prompt) ≈ 1.1 * parse(Float64, base.pricing.prompt)
        end
    end

    @testset "anthropic_eu: Bedrock EU inference profile, anthropic pricing +10%" begin
        @test OpenRouter.bedrock_eu_model_transform("anthropic/claude-haiku-5.5") == "eu.anthropic.claude-haiku-5-5"
        @test OpenRouter.bedrock_eu_model_transform("anthropic/claude-haiku-4.5") == "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        info, model_id, ep = OpenRouter.parse_provider_model("anthropic_eu:anthropic/claude-haiku-4.5")
        _, _, base = OpenRouter.parse_provider_model("anthropic:anthropic/claude-haiku-4.5")
        @test info.base_url == "https://bedrock-runtime.eu-central-1.amazonaws.com/anthropic"
        @test info.schema isa OpenRouter.AnthropicSchema
        @test model_id == "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        @test parse(Float64, ep.pricing.prompt) ≈ 1.1 * parse(Float64, base.pricing.prompt)
        @test info.auth_header_format == "x-api-key"
        @test info.default_headers["anthropic-version"] == "2023-06-01"
        _, sfx_id, _ = OpenRouter.parse_provider_model("anthropic_eu:anthropic/claude-haiku-4.5(high)")
        @test sfx_id == "eu.anthropic.claude-haiku-4-5-20251001-v1:0"  # suffix dropped, Bedrock rejects it
    end

    @testset "openrouter_eu: EU in-region gateway, passthrough id, data_collection deny" begin
        info, mid, ep = OpenRouter.parse_provider_model("openrouter_eu:openai/gpt-6.1-sol")
        @test info.base_url == "https://eu.openrouter.ai/api/v1"
        @test mid == "openai/gpt-6.1-sol"
        @test info.api_key_env_var == "OPENROUTER_EU_API_KEY"
        @test OpenRouter.is_openrouter_gateway(info)
        @test OpenRouter.is_openrouter_gateway(OpenRouter.get_provider_info("openrouter"))
        @test !OpenRouter.is_openrouter_gateway(OpenRouter.get_provider_info("openai"))
    end

    @testset "vertex_eu: Vertex EU multi-region rawPredict, anthropic pricing +10%" begin
        @test OpenRouter.vertex_model_transform("anthropic/claude-haiku-4.5") == "claude-haiku-4-5"
        @test OpenRouter.vertex_model_transform("anthropic/claude-opus-4.8") == "claude-opus-4-8"
        withenv("GOOGLE_CLOUD_PROJECT" => "proj-x") do
            info, model_id, ep = OpenRouter.parse_provider_model("vertex_eu:anthropic/claude-haiku-4.5(high)")
            _, _, base = OpenRouter.parse_provider_model("anthropic:anthropic/claude-haiku-4.5")
            @test model_id == "claude-haiku-4-5"  # suffix dropped
            @test parse(Float64, ep.pricing.prompt) ≈ 1.1 * parse(Float64, base.pricing.prompt)
            root = OpenRouter.resolve_base_url(info.base_url)
            @test OpenRouter.build_url(info.schema, root, model_id) ==
                  "https://aiplatform.eu.rep.googleapis.com/v1/projects/proj-x/locations/eu/publishers/anthropic/models/claude-haiku-4-5:rawPredict"
            @test endswith(OpenRouter.build_url(info.schema, root, model_id, true), ":streamRawPredict")
            p = OpenRouter.build_payload(info.schema, "hi", model_id, nothing, true)
            @test !haskey(p, "model") && p["anthropic_version"] == "vertex-2023-10-16" && p["stream"] == true
        end
        @test haskey(OpenRouter.build_payload(OpenRouter.AnthropicSchema(), "hi", "m", nothing), "model")
        fx = joinpath(@__DIR__, "fixtures")
        jwt = OpenRouter.google_jwt(Dict("client_email" => "a@b", "private_key" => read(joinpath(fx, "test_rsa.pem"), String)), 100)
        h, c, sig = split(jwt, ".")
        unb64(x) = OpenRouter.base64decode(x * repeat("=", mod(-length(x), 4)) |> y -> replace(y, '-' => '+', '_' => '/'))
        claims = OpenRouter.JSON3.read(unb64(c))
        @test claims.iss == "a@b" && claims.iat == 100 && claims.exp == 3700 && claims.aud == OpenRouter.GOOGLE_TOKEN_URI
        pub = OpenRouter.MbedTLS.PKContext()
        OpenRouter.MbedTLS.parse_public_key!(pub, read(joinpath(fx, "test_rsa.pub"), String))
        @test OpenRouter.MbedTLS.verify(pub, OpenRouter.MbedTLS.MD_SHA256,
            OpenRouter.MbedTLS.digest(OpenRouter.MbedTLS.MD_SHA256, "$h.$c"), unb64(sig)) == 0
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

    @testset "reasoning_tokens: subset (OpenAI) vs exclusive (xAI) completion" begin
        cc = OpenRouter.ChatCompletionSchema()
        # xAI grok-4.20-reasoning real usage: billed 0.00121435 USD (cost_in_usd_ticks 12143500)
        xai = Dict("usage" => Dict("prompt_tokens" => 193, "completion_tokens" => 1, "total_tokens" => 636,
                   "prompt_tokens_details" => Dict("cached_tokens" => 128),
                   "completion_tokens_details" => Dict("reasoning_tokens" => 442)))
        t = OpenRouter.extract_tokens(cc, xai)
        @test t.completion_tokens == 443 && t.internal_reasoning == 442 && t.total_tokens == 636
        p = Pricing(prompt="0.00000125", completion="0.0000025", request=nothing, image=nothing,
                    web_search=nothing, internal_reasoning=nothing, image_output=nothing, audio=nothing,
                    input_audio_cache=nothing, input_cache_read="0.0000002", input_cache_write=nothing, discount=nothing)
        @test isapprox(calculate_cost(p, t), 0.00121435; atol=1e-12)
        # OpenAI: reasoning already inside completion_tokens
        oai = Dict("usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 500, "total_tokens" => 510,
                   "completion_tokens_details" => Dict("reasoning_tokens" => 400)))
        t2 = OpenRouter.extract_tokens(cc, oai)
        @test t2.completion_tokens == 500 && t2.internal_reasoning == 400
    end

    @testset "Gemini: thoughts folded, cached input split, fallback total" begin
        g = OpenRouter.GeminiSchema()
        t = OpenRouter.extract_tokens(g, Dict("usageMetadata" => Dict("promptTokenCount" => 100, "cachedContentTokenCount" => 60,
                                                                       "candidatesTokenCount" => 2, "thoughtsTokenCount" => 20)))
        @test t.prompt_tokens == 40 && t.input_cache_read == 60
        @test t.completion_tokens == 22 && t.internal_reasoning == 20 && t.total_tokens == 122
    end

    @testset "Anthropic stream acc_tokens keeps input/cache from message_start" begin
        s = OpenRouter.AnthropicSchema()
        start = OpenRouter.extract_tokens(s, Dict("type"=>"message_start", "message"=>Dict("usage"=>Dict(
            "input_tokens"=>44, "output_tokens"=>1, "cache_read_input_tokens"=>200, "cache_creation_input_tokens"=>50))))
        delta = OpenRouter.extract_tokens(s, Dict("type"=>"message_delta", "usage"=>Dict("output_tokens"=>24)))
        acc = OpenRouter.acc_tokens(s, start, delta)
        @test acc.prompt_tokens == 44 && acc.input_cache_read == 200 && acc.input_cache_write == 50
        @test acc.completion_tokens == 24 && acc.total_tokens == 44 + 200 + 50 + 24
    end

    @testset "pricing.discount is informational (prices already net)" begin
        # OpenRouter: Mistral Large 4 lists $0.68/M with discount=0.5 (list $1.36); must not halve again.
        mk(d) = Pricing(prompt="0.00000068", completion="0.00000209", request=nothing, image=nothing,
                        web_search=nothing, internal_reasoning=nothing, image_output=nothing, audio=nothing,
                        input_audio_cache=nothing, input_cache_read="0.00000007", input_cache_write=nothing, discount=d)
        tokens = Dict(:prompt_tokens => 9, :completion_tokens => 478)
        expected = 9*0.00000068 + 478*0.00000209
        @test isapprox(calculate_cost(mk(0.5), tokens), expected; atol=1e-12)
        @test calculate_cost(mk(0.5), tokens) == calculate_cost(mk(nothing), tokens)
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

    @testset "Anthropic max_tokens defaults to catalog max output" begin
        using OpenRouter: AnthropicSchema, ChatCompletionSchema, with_default_max_tokens, create_stub_endpoint
        ep = create_stub_endpoint("anthropic", "claude-sonnet-4.5")
        ep.max_completion_tokens = 64000

        # No user max_tokens → catalog max output.
        kw = with_default_max_tokens(AnthropicSchema(), ep, pairs((;)))
        @test Dict(kw)[:max_tokens] == 64000

        # Explicit user max_tokens wins.
        kw = with_default_max_tokens(AnthropicSchema(), ep, pairs((; max_tokens=10)))
        @test Dict(kw)[:max_tokens] == 10

        # Unknown catalog value → untouched (schema fallback applies).
        ep2 = create_stub_endpoint("anthropic", "claude-x")
        kw = with_default_max_tokens(AnthropicSchema(), ep2, pairs((;)))
        @test !haskey(Dict(kw), :max_tokens)

        # Non-Anthropic schemas untouched.
        kw = with_default_max_tokens(ChatCompletionSchema(), ep, pairs((;)))
        @test !haskey(Dict(kw), :max_tokens)
    end

    @testset "Gemini maps max_tokens to generationConfig.maxOutputTokens" begin
        using OpenRouter: GeminiSchema, build_payload
        # Gemini API rejects OpenAI-style `max_tokens` at payload root:
        # 400 Invalid JSON payload: Unknown name "max_tokens".
        for key in (:max_tokens, :max_completion_tokens, :max_output_tokens, :maxOutputTokens)
            p = build_payload(GeminiSchema(), "hi", "gemini-3.7-flash", nothing; (key => 123,)...)
            @test !haskey(p, "max_tokens")
            @test !haskey(p, "max_completion_tokens")
            @test p["generationConfig"]["maxOutputTokens"] == 123
        end
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

    @testset "Streaming error with gzip Content-Encoding" begin
        using OpenRouter: HttpStreamCallback, ChatCompletionSchema, streamed_request!
        import HTTP, CodecZlib

        # Mimic CLIProxyAPI's 403 region-lock: gzip-compressed JSON body. HTTP.open
        # bypasses auto-decompression, so the error path must gunzip it itself,
        # otherwise the user sees raw gzip bytes instead of the real message.
        json = """{"code":"permission-denied","error":"The model grok-4.5 is not available in your region."}"""
        gz = transcode(CodecZlib.GzipCompressor, Vector{UInt8}(json))
        handler(_) = HTTP.Response(403, ["Content-Type" => "application/json",
                                         "Content-Encoding" => "gzip"]; body=gz)
        server = HTTP.serve!(handler, "127.0.0.1", 18798)
        try
            url = "http://127.0.0.1:18798/v1/chat/completions"
            err = try
                streamed_request!(HttpStreamCallback(; schema=ChatCompletionSchema()),
                                  url, ["Content-Type" => "application/json"], "{}")
                nothing
            catch e
                e
            end
            @test err isa HTTP.RequestError
            @test occursin("403", err.error)
            @test occursin("not available in your region", err.error)
        finally
            close(server)
        end
    end

    include("test_endpoint_refresh.jl")

    # Include custom provider tests
    include("test_custom_providers.jl")

    # Tool message serialization tests
    include("test_tool_messages.jl")

    # Ollama native API tests (live cloud tests gated on OLLAMA_API_KEY)
    include("test_ollama_provider.jl")

    # model(level) suffix -> reasoning_effort for native OpenAI-compatible providers (meta)
    include("test_thinking_suffix.jl")

    # Ollama -> OpenRouter catalog price matching (export script)
    include("test_ollama_pricing_match.jl")

    # OpenCode Go native provider + export catalog
    include("test_opencode_go_provider.jl")

    # Echo server tests (all schemas)
    include("test_echo_server.jl")

    # Stream first-chunk/idle timeout tests
    include("test_stream_timeouts.jl")

    # Fast cooperative cancel (stream_cancel_flag) tests
    include("test_stream_cancel.jl")

    # Request payload size reporting (diagnosing max-request-size rejections)
    include("test_request_size.jl")

    # Model refusal (safety stop) -> ModelRefusalError instead of a parse error
    include("test_model_refusal.jl")
    include("test_anthropic_content.jl")

    # ChatCompletion delta.images (CLIProxyAPI Codex image_generation) -> AIMessage.image_data
    include("test_chatcompletion_images.jl")

    # Mistral reasoning models: content as thinking/text block arrays
    include("test_mistral_content_blocks.jl")
    include("test_meta_model_transform.jl")
    include("test_anthropic_hooks_thinking.jl")
    include("test_provenance.jl")
end
