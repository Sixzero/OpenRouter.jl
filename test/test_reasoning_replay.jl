using Test, JSON3
using OpenRouter
using OpenRouter: AbstractMessage, UserMessage, AIMessage, ToolMessage,
    AnthropicSchema, ResponseSchema, ChatCompletionSchema, GeminiSchema,
    StreamChunk, HttpStreamHooks, build_payload, build_response_body,
    extract_reasoning_items, extract_reasoning, to_anthropic_messages, to_gemini_contents

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

    @testset "Gemini: thoughtSignature bound to the producing model" begin
        M = "gemini-3.8-flash-high"
        fc_turn = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [
            Dict("text" => "plan", "thought" => true),
            Dict("functionCall" => Dict("name" => "calc", "args" => Dict("a" => 1), "id" => "c1"), "thoughtSignature" => "SIG_FC"),
        ]))])
        text_turn = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [
            Dict("text" => "Answer: 3", "thoughtSignature" => "SIG_TXT"),
        ]))])
        a1 = AIMessage(GeminiSchema(), fc_turn; model_id=M)
        a2 = AIMessage(GeminiSchema(), text_turn; model_id=M)
        @test a1.reasoning_model == M && a1.tool_calls[1]["thoughtSignature"] == "SIG_FC"
        @test [i["part"] for i in a1.reasoning_items] == ["functionCall"]
        @test a2.reasoning_items[1]["signature"] == "SIG_TXT"
        msgs = AbstractMessage[UserMessage(content="q"), a1, ToolMessage(content="1", tool_call_id="c1", name="calc"), a2, UserMessage(content="again")]
        parts(model) = [c["parts"] for c in to_gemini_contents(msgs; model_id=model) if c["role"] == "model"]
        same = parts(M)
        @test same[1][end]["thoughtSignature"] == "SIG_FC"
        @test same[2][end]["thoughtSignature"] == "SIG_TXT"
        @test parts(M * "(low)")[1][end]["thoughtSignature"] == "SIG_FC"     # level suffix: same model
        other = parts("gemini-3.1-flash-lite")
        @test other[1][end]["thoughtSignature"] == OpenRouter.GEMINI_SKIP_SIGNATURE
        @test !haskey(other[2][end], "thoughtSignature")
        # Foreign/unbound history (e.g. a Claude tool call) still gets the sentinel
        foreign = AIMessage(content="", tool_calls=[Dict{String,Any}("id" => "t1", "type" => "function", "function" => Dict("name" => "calc", "arguments" => "{}"))])
        fp = to_gemini_contents(AbstractMessage[UserMessage(content="q"), foreign]; model_id=M)[2]["parts"]
        @test fp[1]["thoughtSignature"] == OpenRouter.GEMINI_SKIP_SIGNATURE
        unnamed = AbstractMessage[UserMessage(content="q"), a1, ToolMessage(content="1", tool_call_id="c1")]
        @test to_gemini_contents(unnamed)[3]["parts"][1]["functionResponse"]["name"] == "calc"   # resolved from the call
        @test !haskey(only(build_payload(ChatCompletionSchema(), msgs, "x/y", nothing)["messages"][2]["tool_calls"]), "thoughtSignature")
        @test build_payload(GeminiSchema(), msgs, M, nothing)["contents"][2]["parts"][end]["thoughtSignature"] == "SIG_FC"
    end

    @testset "Gemini streaming keeps the text-part signature" begin
        events = [
            """{"candidates":[{"content":{"role":"model","parts":[{"text":"hmm","thought":true}]}}]}""",
            """{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer"}]}}]}""",
            """{"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"SIG_STREAM"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}}""",
        ]
        cb = HttpStreamHooks(schema=GeminiSchema(), out=devnull, reasoning_formatter=_ -> nothing, content_formatter=_ -> nothing)
        for e in events
            c = StreamChunk(data=e, json=JSON3.read(e)); push!(cb, c); OpenRouter.callback(cb, c)
        end
        body = JSON3.read(JSON3.write(build_response_body(GeminiSchema(), cb)), Dict)
        msg = AIMessage(GeminiSchema(), body; model_id="gemini-3.8-flash-high")
        @test msg.content == "Answer"
        @test msg.reasoning_items == Any[Dict{String,Any}("type" => "thought_signature", "signature" => "SIG_STREAM", "part" => "text")]
    end


    @testset "Gemini: parallel calls, ids, multi-part text, named tool_choice" begin
        M = "gemini-3.8-flash-high"
        par = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [
            Dict("text" => "A"), Dict("text" => "B"),
            Dict("functionCall" => Dict("name" => "calc", "args" => Dict("a" => 1), "id" => "c1"), "thoughtSignature" => "SIG1"),
            Dict("functionCall" => Dict("name" => "calc", "args" => Dict("a" => 2), "id" => "c2")),
        ]))])
        a = AIMessage(GeminiSchema(), par; model_id=M)
        @test a.content == "AB"
        msgs = AbstractMessage[UserMessage(content="q"), a,
            ToolMessage(content="2", tool_call_id="c2"), ToolMessage(content="1", tool_call_id="c1")]
        sigs(model) = [get(p, "thoughtSignature", nothing) for p in to_gemini_contents(msgs; model_id=model)[2]["parts"] if haskey(p, "functionCall")]
        @test sigs(M) == ["SIG1", nothing]                                        # sibling stays unsigned
        @test sigs("other") == [OpenRouter.GEMINI_SKIP_SIGNATURE, nothing]        # sentinel on the first only
        c = to_gemini_contents(msgs; model_id=M)
        @test [p["functionCall"]["id"] for p in c[2]["parts"] if haskey(p, "functionCall")] == ["c1", "c2"]
        @test [p["functionResponse"]["id"] for p in c[3]["parts"]] == ["c2", "c1"]
        tc = Dict("type" => "function", "function" => Dict("name" => "calc"))
        @test build_payload(GeminiSchema(), "hi", M, nothing; tool_choice=tc)["toolConfig"]["functionCallingConfig"] ==
              Dict("mode" => "ANY", "allowedFunctionNames" => ["calc"])
        @test_throws ArgumentError build_payload(GeminiSchema(), "hi", M, nothing; tool_choice="bogus")
        noargs = Dict("candidates" => [Dict("content" => Dict("parts" => [Dict("functionCall" => Dict("name" => "ping"))]))])
        @test AIMessage(GeminiSchema(), noargs; model_id=M).tool_calls[1]["function"]["arguments"] == "{}"
        ev = """{"candidates":[{"content":{"parts":[{"text":"A"},{"text":"B"}]}}]}"""
        @test OpenRouter.extract_content(GeminiSchema(), StreamChunk(data=ev, json=JSON3.read(ev))) == "AB"
    end

end
