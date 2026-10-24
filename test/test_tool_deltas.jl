using Test, JSON3, OpenRouter
using OpenRouter: AnthropicSchema, ChatCompletionSchema, ResponseSchema, StreamChunk

function collect_tool_deltas(schema, raws)
    seen = Tuple[]
    cb = OpenRouter.HttpStreamHooks(schema=schema, out=devnull,
        on_tool_delta = (i, id, name, args) -> push!(seen, (i, id, name, args)))
    for r in raws
        OpenRouter.callback(cb, StreamChunk(data=r, json=JSON3.read(r)))
    end
    seen
end

@testset "HttpStreamHooks: on_tool_delta" begin
    @test collect_tool_deltas(AnthropicSchema(), [
        """{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash","input":{}}}""",
        """{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"cmd\\": \\"ls"}}""",
        """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}""",
    ]) == [(1, "toolu_1", "bash", ""), (1, nothing, nothing, "{\"cmd\": \"ls")]

    @test collect_tool_deltas(ChatCompletionSchema(), [
        """{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]}}]}""",
        """{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"cmd\\""}}]}}]}""",
        """{"choices":[{"index":0,"delta":{"content":"x"}}]}""",
    ]) == [(0, "call_1", "bash", ""), (0, nothing, nothing, "{\"cmd\"")]

    @test collect_tool_deltas(ResponseSchema(), [
        """{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_9","name":"read","arguments":""}}""",
        """{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\\"path"}""",
    ]) == [(2, "call_9", "read", ""), (2, nothing, nothing, "{\"path")]
end
