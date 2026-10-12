using Test, OpenRouter
using OpenRouter: AIMessage, AnthropicSchema, ModelRefusalError, extract_content

@testset "Anthropic response content" begin
    result = Dict{String,Any}(
        "model" => "claude-opus-5-5", "role" => "assistant",
        "content" => Any[], "stop_reason" => "end_turn", "stop_details" => nothing,
        "usage" => Dict("input_tokens" => 2, "output_tokens" => 2),
    )

    @testset "Empty end_turn" begin
        msg = @test_logs (:warn, "Model response has no text and no tool calls") AIMessage(AnthropicSchema(), result)
        @test msg.content == ""
        @test msg.finish_reason == "end_turn"
        @test msg.tokens.prompt_tokens == 2
        @test msg.tokens.completion_tokens == 2
        @test msg.tool_calls === nothing
    end

    @testset "Text and tool-only responses" begin
        text = merge(result, Dict("content" => [Dict("type" => "text", "text" => "hi")]))
        @test AIMessage(AnthropicSchema(), text).content == "hi"

        tool = merge(result, Dict("stop_reason" => "tool_use", "content" => [
            Dict("type" => "tool_use", "id" => "toolu_test", "name" => "html_snippet",
                 "input" => Dict("html" => "<p>Connect GitHub</p>")),
        ]))
        msg = AIMessage(AnthropicSchema(), tool)
        @test msg.content == ""
        @test length(msg.tool_calls) == 1
        @test msg.tool_calls[1]["function"]["name"] == "html_snippet"
    end

    @testset "Refusals and malformed responses still fail" begin
        refusal = merge(result, Dict("stop_reason" => "refusal",
            "stop_details" => Dict("type" => "refusal", "category" => "bio")))
        @test_throws ModelRefusalError AIMessage(AnthropicSchema(), refusal)
        missing_content = copy(result)
        delete!(missing_content, "content")
        @test_throws ErrorException extract_content(AnthropicSchema(), missing_content)
        for content in (nothing, "", Dict())
            @test_throws ErrorException extract_content(AnthropicSchema(), merge(result, Dict("content" => content)))
        end
    end
end
