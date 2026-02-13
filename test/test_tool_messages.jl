using Test
using OpenRouter: AbstractMessage, SystemMessage, UserMessage, AIMessage, ToolMessage,
    to_openai_messages, to_anthropic_messages, to_gemini_contents
using OpenRouter: ResponseSchema, build_payload

# Shared: user → AI calls 2 tools → tool results → AI responds
function tool_conversation()
    AbstractMessage[
        SystemMessage(content="You are helpful."),
        UserMessage(content="What's the weather in NYC and SF?"),
        AIMessage(content="Let me check both cities.",
            tool_calls=[
                Dict("id" => "call_1", "type" => "function",
                     "function" => Dict("name" => "get_weather", "arguments" => """{"city":"New York"}""")),
                Dict("id" => "call_2", "type" => "function",
                     "function" => Dict("name" => "get_weather", "arguments" => """{"city":"San Francisco"}"""))
            ]),
        ToolMessage(content="72°F, sunny", tool_call_id="call_1", name="get_weather"),
        ToolMessage(content="58°F, foggy", tool_call_id="call_2", name="get_weather"),
        AIMessage(content="NYC is 72°F sunny, SF is 58°F foggy.")
    ]
end

@testset "Tool Messages" begin
    msgs = tool_conversation()

    @testset "OpenAI" begin
        out = to_openai_messages(msgs)
        @test length(out) == 6
        @test out[1]["role"] == "system"
        @test out[2]["role"] == "user"
        # AI with tool_calls
        @test out[3]["role"] == "assistant"
        @test out[3]["content"] == "Let me check both cities."
        @test length(out[3]["tool_calls"]) == 2
        @test out[3]["tool_calls"][1]["id"] == "call_1"
        # Separate tool messages
        @test out[4] == Dict("role" => "tool", "tool_call_id" => "call_1", "content" => "72°F, sunny")
        @test out[5] == Dict("role" => "tool", "tool_call_id" => "call_2", "content" => "58°F, foggy")
        @test out[6]["role"] == "assistant"
    end

    @testset "Anthropic" begin
        out = to_anthropic_messages(msgs)
        # SystemMessage skipped → user, assistant(text+2 tool_use), user(2 tool_result), assistant
        @test length(out) == 4

        @test out[1]["role"] == "user"

        # AI: text + 2 tool_use blocks
        ai = out[2]
        @test ai["role"] == "assistant"
        @test ai["content"][1] == Dict{String,Any}("type" => "text", "text" => "Let me check both cities.")
        @test length(ai["content"]) == 3
        @test ai["content"][2]["type"] == "tool_use"
        @test ai["content"][2]["id"] == "call_1"
        @test ai["content"][2]["name"] == "get_weather"
        @test ai["content"][2]["input"] == Dict{String,Any}("city" => "New York")
        @test ai["content"][3]["id"] == "call_2"
        @test ai["content"][3]["input"] == Dict{String,Any}("city" => "San Francisco")

        # Consecutive ToolMessages grouped into one user message
        tr = out[3]
        @test tr["role"] == "user"
        @test length(tr["content"]) == 2
        @test tr["content"][1]["type"] == "tool_result"
        @test tr["content"][1]["tool_use_id"] == "call_1"
        @test tr["content"][1]["content"] == "72°F, sunny"
        @test tr["content"][2]["tool_use_id"] == "call_2"
        @test tr["content"][2]["content"] == "58°F, foggy"

        @test out[4]["role"] == "assistant"
    end

    @testset "Gemini" begin
        out = to_gemini_contents(msgs)
        # SystemMessage skipped → user, model(text+2 functionCall), user(2 functionResponse), model
        @test length(out) == 4

        @test out[1]["role"] == "user"

        # Model: text + 2 functionCall parts
        model = out[2]
        @test model["role"] == "model"
        @test model["parts"][1] == Dict("text" => "Let me check both cities.")
        @test length(model["parts"]) == 3
        @test model["parts"][2]["functionCall"]["name"] == "get_weather"
        @test model["parts"][2]["functionCall"]["args"] == Dict{String,Any}("city" => "New York")
        @test model["parts"][3]["functionCall"]["args"] == Dict{String,Any}("city" => "San Francisco")

        # Consecutive ToolMessages grouped
        tr = out[3]
        @test tr["role"] == "user"
        @test length(tr["parts"]) == 2
        @test tr["parts"][1]["functionResponse"]["name"] == "get_weather"
        @test tr["parts"][1]["functionResponse"]["response"] == Dict{String,Any}("content" => "72°F, sunny")
        @test tr["parts"][2]["functionResponse"]["response"] == Dict{String,Any}("content" => "58°F, foggy")

        @test out[4]["role"] == "model"
    end

    @testset "Single ToolMessage" begin
        single = AbstractMessage[
            UserMessage(content="Hi"),
            AIMessage(content="Calling.", tool_calls=[
                Dict("id" => "tc_1", "type" => "function",
                     "function" => Dict("name" => "foo", "arguments" => "{}"))]),
            ToolMessage(content="result", tool_call_id="tc_1"),
            AIMessage(content="Done.")
        ]
        anth = to_anthropic_messages(single)
        @test length(anth) == 4
        @test length(anth[3]["content"]) == 1  # single tool_result, not grouped

        gem = to_gemini_contents(single)
        @test length(gem) == 4
        @test length(gem[3]["parts"]) == 1
    end

    @testset "Anthropic cache with tool results" begin
        out = to_anthropic_messages(msgs; cache=:last)
        # Last user message is the tool_results one
        @test out[3]["role"] == "user"
        @test haskey(out[3]["content"][end], "cache_control")
    end

    @testset "ResponseSchema" begin
        payload = build_payload(ResponseSchema(), msgs, "gpt-5", nothing, false)
        input = payload["input"]
        @test payload["instructions"] == "You are helpful."

        # user message
        @test input[1]["type"] == "message"
        @test input[1]["role"] == "user"

        # AI with content + tool_calls → assistant message + 2 function_call items
        @test input[2]["type"] == "message"
        @test input[2]["role"] == "assistant"
        @test input[2]["content"] == "Let me check both cities."
        @test input[3]["type"] == "function_call"
        @test input[3]["call_id"] == "call_1"
        @test input[3]["name"] == "get_weather"
        @test input[4]["type"] == "function_call"
        @test input[4]["call_id"] == "call_2"

        # ToolMessages → function_call_output
        @test input[5]["type"] == "function_call_output"
        @test input[5]["call_id"] == "call_1"
        @test input[5]["output"] == "72°F, sunny"
        @test input[6]["type"] == "function_call_output"
        @test input[6]["call_id"] == "call_2"

        # Final assistant
        @test input[7]["type"] == "message"
        @test input[7]["role"] == "assistant"
    end

    @testset "AIMessage without tool_calls unchanged" begin
        plain = AbstractMessage[
            UserMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        anth = to_anthropic_messages(plain)
        @test length(anth) == 2
        @test anth[2]["content"] == [Dict{String,Any}("type" => "text", "text" => "Hi there!")]

        gem = to_gemini_contents(plain)
        @test length(gem) == 2
        @test gem[2]["parts"] == Any[Dict("text" => "Hi there!")]
    end
end
