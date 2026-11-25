using Test, JSON3
using OpenRouter
using OpenRouter: AbstractMessage, UserMessage, AIMessage, ToolMessage,
    AnthropicSchema, ResponseSchema, ChatCompletionSchema,
    StreamChunk, HttpStreamHooks, build_payload, build_response_body,
    extract_reasoning_items, extract_reasoning, to_anthropic_messages

@testset "Reasoning state extraction & replay" begin
    @testset "Anthropic: extract signed + redacted, drop unsigned" begin
        raw = Dict{String,Any}(
            "content" => Any[
                Dict{String,Any}("type" => "thinking", "thinking" => "step 1", "signature" => "sig_a"),
                Dict{String,Any}("type" => "redacted_thinking", "data" => "opaque_b"),
                Dict{String,Any}("type" => "thinking", "thinking" => "", "signature" => "sig_empty_text"),
                Dict{String,Any}("type" => "thinking", "thinking" => "truncated without sig"),
                Dict{String,Any}("type" => "text", "text" => "answer"),
            ],
            "stop_reason" => "end_turn",
        )
        msg = AIMessage(AnthropicSchema(), raw; model_id="claude-opus-4-6(xhigh)")
        @test msg.reasoning == "step 1"
        @test msg.reasoning_model == "claude-opus-4-6(xhigh)"
        @test length(msg.reasoning_items) == 3
        @test msg.reasoning_items[1] == Dict("type" => "thinking", "thinking" => "step 1", "signature" => "sig_a")
        @test msg.reasoning_items[2] == Dict("type" => "redacted_thinking", "data" => "opaque_b")
        @test msg.reasoning_items[3] == Dict("type" => "thinking", "thinking" => "", "signature" => "sig_empty_text")
    end

    @testset "Anthropic streaming captures signature_delta and redacted_thinking" begin
        events = [
            """{"type":"message_start","message":{"id":"m1","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}""",
            """{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"check"}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_part1_"}}""",
            """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"part2"}}""",
            """{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"red_xyz"}}""",
            """{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}""",
            """{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"cmd\\":\\"ls\\"}"}}""",
            """{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}""",
        ]
        cb = HttpStreamHooks(schema=AnthropicSchema(), out=devnull,
                             reasoning_formatter=_ -> nothing, content_formatter=_ -> nothing)
        for e in events
            c = StreamChunk(data=e, json=JSON3.read(e)); push!(cb, c); OpenRouter.callback(cb, c)
        end
        body = JSON3.read(JSON3.write(build_response_body(AnthropicSchema(), cb)), Dict)
        msg = AIMessage(AnthropicSchema(), body; model_id="claude-opus-4-6(high)")
        @test msg.reasoning == "let me check"
        @test length(msg.reasoning_items) == 2
        @test msg.reasoning_items[1]["signature"] == "sig_part1_part2"
        @test msg.reasoning_items[2] == Dict("type" => "redacted_thinking", "data" => "red_xyz")
        @test msg.tool_calls !== nothing && length(msg.tool_calls) == 1

        # Replay to same model (even different thinking suffix): thinking + redacted precede tool_use, no empty text
        msgs = AbstractMessage[
            UserMessage(content="run ls"),
            msg,
            ToolMessage(content="a.txt", tool_call_id="toolu_1"),
        ]
        p_same = build_payload(AnthropicSchema(), msgs, "claude-opus-4-6(medium)", nothing, false)
        ai_content = p_same["messages"][2]["content"]
        @test [b["type"] for b in ai_content] == ["thinking", "redacted_thinking", "tool_use"]
        @test ai_content[1]["signature"] == "sig_part1_part2"
        @test ai_content[2]["data"] == "red_xyz"

        # Replay to different model: opaque blocks dropped
        p_diff = build_payload(AnthropicSchema(), msgs, "claude-sonnet-4-6(high)", nothing, false)
        @test [b["type"] for b in p_diff["messages"][2]["content"]] == ["tool_use"]
    end

    @testset "Responses API: include encrypted_content, extract & replay without id" begin
        raw = Dict{String,Any}(
            "status" => "completed",
            "output" => Any[
                Dict{String,Any}(
                    "type" => "reasoning",
                    "id" => "rs_123",
                    "summary" => Any[Dict{String,Any}("type" => "summary_text", "text" => "Checked 17*23")],
                    "encrypted_content" => "gAAAA_encrypted_blob",
                ),
                Dict{String,Any}(
                    "type" => "reasoning",
                    "id" => "rs_no_enc",
                    "summary" => Any[],
                ),
                Dict{String,Any}(
                    "type" => "function_call",
                    "call_id" => "fc_1",
                    "name" => "calc",
                    "arguments" => """{"x":1}""",
                ),
            ],
        )
        msg = AIMessage(ResponseSchema(), raw; model_id="gpt-5.6-luna")
        @test msg.reasoning == "Checked 17*23"
        @test msg.reasoning_model == "gpt-5.6-luna"
        @test length(msg.reasoning_items) == 1
        @test msg.reasoning_items[1]["encrypted_content"] == "gAAAA_encrypted_blob"
        @test !haskey(msg.reasoning_items[1], "id")

        msgs = AbstractMessage[
            UserMessage(content="compute"),
            msg,
            ToolMessage(content="391", tool_call_id="fc_1"),
        ]
        p = build_payload(ResponseSchema(), msgs, "gpt-5.6-luna", nothing, false)
        @test p["include"] == ["reasoning.encrypted_content"]
        # input: user message, reasoning item, function_call, function_call_output
        @test [i["type"] for i in p["input"]] == ["message", "reasoning", "function_call", "function_call_output"]
        @test p["input"][2]["encrypted_content"] == "gAAAA_encrypted_blob"
        @test !haskey(p["input"][2], "id")

        # Model switch drops opaque reasoning
        p_other = build_payload(ResponseSchema(), msgs, "gpt-5-mini", nothing, false)
        @test [i["type"] for i in p_other["input"]] == ["message", "function_call", "function_call_output"]

        # Cross-schema guard: Responses item never injected into Anthropic payload even with same model_id
        p_cross = build_payload(AnthropicSchema(), msgs, "gpt-5.6-luna", nothing, false)
        @test [b["type"] for b in p_cross["messages"][2]["content"]] == ["tool_use"]

        # store / include kwargs forwarded cleanly
        p_custom = build_payload(ResponseSchema(), "hi", "gpt-5.6-luna", nothing, false; store=false)
        @test p_custom["store"] === false
    end
end
