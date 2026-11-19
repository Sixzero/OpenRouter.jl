using Test
using OpenRouter
using OpenRouter: AIMessage, AnthropicSchema, ChatCompletionSchema, GeminiSchema, ModelRefusalError

@testset "Model refusal" begin
    @testset "Anthropic contentless refusal" begin
        result = Dict(
            "model" => "claude-fable-5", "role" => "assistant", "content" => Any[],
            "stop_reason" => "refusal",
            "stop_details" => Dict("type" => "refusal", "category" => "bio"),
            "usage" => Dict("input_tokens" => 2, "output_tokens" => 0),
        )
        e = try AIMessage(AnthropicSchema(), result); nothing catch err; err end
        @test e isa ModelRefusalError
        msg = sprint(showerror, e)
        @test occursin("refused", msg)
        @test occursin("claude-fable-5", msg)
        @test occursin("bio", msg)
        @test !occursin("Unexpected response format", msg)
    end

    @testset "OpenAI content_filter" begin
        result = Dict("model" => "gpt-x", "usage" => Dict(),
            "choices" => [Dict("finish_reason" => "content_filter",
                               "message" => Dict("content" => ""))])
        @test_throws ModelRefusalError AIMessage(ChatCompletionSchema(), result)
    end

    @testset "Gemini SAFETY" begin
        result = Dict("model" => "gemini-x",
            "candidates" => [Dict("finishReason" => "SAFETY", "content" => Dict("parts" => []))])
        @test_throws ModelRefusalError AIMessage(GeminiSchema(), result)
    end

    @testset "OpenAI HTTP 400 cyber flag" begin
        body = """{"error":{"message":"This content was flagged for possible cybersecurity risk. If this seems wrong, try rephrasing your request.","type":"invalid_request_error","code":null}}"""
        e = OpenRouter.refusal_from_http_error(400, body, """{"model":"gpt-6.1-sol","stream":true}""")
        @test e isa ModelRefusalError
        @test e.model == "gpt-6.1-sol"
        @test e.category == "cyber"
        @test OpenRouter.refusal_from_http_error(400, """{"error":{"code":"content_filter"}}""", "{}") isa ModelRefusalError
        @test OpenRouter.refusal_from_http_error(400, """{"error":{"message":"context length exceeded"}}""", "{}") === nothing
        @test OpenRouter.refusal_from_http_error(429, body, "{}") === nothing
    end

    @testset "CLIProxyAPI policy-block renderings" begin
        req = """{"model":"claude-opus-5-5"}"""
        r(body) = OpenRouter.refusal_from_http_error(400, body, req)
        e = r("""{"type":"error","error":{"type":"invalid_request_error","message":"claude refusal (cyber): The request was declined."}}""")
        @test e isa ModelRefusalError && e.model == "claude-opus-5-5" && e.category == "cyber"
        @test r("""{"error":{"message":"claude refusal"}}""").category === nothing
        @test r("""{"error":{"code":"bio_policy","message":"This content was flagged for possible biological risk."}}""").category == "bio"
        @test r("""{"error":{"code":"cyber_policy","message":"x"}}""").category == "cyber"
        @test r("""{"error":{"code":"invalid_prompt","message":"Invalid prompt: we've limited access to this content for safety reasons."}}""") isa ModelRefusalError
        @test r("""{"error":{"message":"This request was blocked due to a misalignment policy violation."}}""").category == "misalignment"
        @test r("""{"error":{"message":"This request was blocked by Gemini's filters. https://policies.google.com/terms/generative-ai/use-policy"}}""") isa ModelRefusalError
        @test r("""{"error":{"code":"invalid_prompt","message":"Invalid request."}}""") === nothing
        @test r("""{"error":{"message":"Invalid field: claude refusal (cyber)","input":{"code":"bio_policy"}}}""") === nothing
        @test r("""{"error":{"code":"image_content_policy_violation"}}""") isa ModelRefusalError
    end

    @testset "In-stream error event" begin
        S = OpenRouter.StreamChunk
        refusal = S(event=nothing, data="""{"error":{"message":"claude refusal (bio): no"}}""",
                    json=OpenRouter.JSON3.read("""{"error":{"message":"claude refusal (bio): no"}}"""))
        e = try OpenRouter.handle_error_message(refusal; request="""{"model":"m1"}""") catch err err end
        @test e isa ModelRefusalError && e.model == "m1" && e.category == "bio"
        other = S(event=:error, data="""{"error":{"message":"overloaded"}}""", json=nothing)
        e2 = try OpenRouter.handle_error_message(other) catch err err end
        @test e2 isa ErrorException
        @test occursin("overloaded", e2.msg)
    end

    @testset "Streamed proxy refusals reach the caller as ModelRefusalError" begin
        import HTTP
        using OpenRouter: HttpStreamCallback, ChatCompletionSchema, streamed_request!
        # HTTP 400 (CLIProxyAPI Claude refusal) and HTTP 200 + in-stream error (Codex policy block).
        bodies = Dict(
            "/400" => HTTP.Response(400, ["Content-Type" => "application/json"];
                body="""{"type":"error","error":{"type":"invalid_request_error","message":"claude refusal (cyber): declined"}}"""),
            "/sse" => HTTP.Response(200, ["Content-Type" => "text/event-stream"];
                body="data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"error\":{\"code\":\"bio_policy\",\"message\":\"This content was flagged for possible biological risk.\"}}\n\n"),
            "/failed" => HTTP.Response(200, ["Content-Type" => "text/event-stream"];
                body="event: response.failed\ndata: {\"response\":{\"error\":{\"code\":\"cyber_policy\"}}}\n\n"))
        server = HTTP.serve!(req -> get(bodies, req.target, bodies["/400"]), "127.0.0.1", 18797)
        base = "http://127.0.0.1:18797"
        try
            for (path, category) in ("/400" => "cyber", "/sse" => "bio", "/failed" => "cyber")
                err = try
                    streamed_request!(HttpStreamCallback(; schema=ChatCompletionSchema(), out=devnull),
                        "$base$path", ["Content-Type" => "application/json"], """{"model":"m-$path"}""")
                    nothing
                catch e
                    e
                end
                @test err isa ModelRefusalError
                @test err isa ModelRefusalError && err.model == "m-$path" && err.category == category
            end
            provider = OpenRouter.ProviderInfo(base, "Bearer", nothing, Dict{String,String}(), nothing,
                ChatCompletionSchema(), "mock proxy")
            endpoint = OpenRouter.create_stub_endpoint("mock", "test-model")
            @test_throws ModelRefusalError OpenRouter._aigen_core("hi", provider, "test-model", endpoint)
        finally
            close(server)
        end
    end

    @testset "Normal responses are unaffected" begin
        ok = AIMessage(ChatCompletionSchema(), Dict("model" => "gpt-x", "usage" => Dict(),
            "choices" => [Dict("finish_reason" => "stop", "message" => Dict("content" => "hi"))]))
        @test ok.content == "hi"
        @test ok.finish_reason == "stop"
    end
end
