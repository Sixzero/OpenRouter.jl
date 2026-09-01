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

    @testset "Normal responses are unaffected" begin
        ok = AIMessage(ChatCompletionSchema(), Dict("model" => "gpt-x", "usage" => Dict(),
            "choices" => [Dict("finish_reason" => "stop", "message" => Dict("content" => "hi"))]))
        @test ok.content == "hi"
        @test ok.finish_reason == "stop"
    end
end
