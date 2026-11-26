using OpenRouter
using OpenRouter: ChatCompletionSchema, StreamChunk, HttpStreamCallback, build_response_body, extract_tool_deltas
using JSON3
using Test

# OpenCode Go hungerg-free repeats `"id":"","type":"","name":""` on every continuation chunk
# (real capture 2026-11-26); the first chunk's id/name must survive.
@testset "ChatCompletion tool stream: empty continuation fields" begin
    raws = [
        """{"choices":[{"index":0,"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]}}]}""",
        """{"choices":[{"index":0,"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"","type":"","function":{"name":"","arguments":"{\\"cmd\\": \\"ls\\""}}]}}]}""",
        """{"choices":[{"index":0,"delta":{"role":"assistant","content":"","tool_calls":[{"index":0,"id":"","type":"","function":{"name":"","arguments":"}"}}]}}]}""",
        """{"choices":[{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}""",
    ]
    cb = HttpStreamCallback()
    for r in raws
        push!(cb.chunks, StreamChunk(data=r, json=JSON3.read(r)))
    end
    schema = ChatCompletionSchema()
    tc = JSON3.read(JSON3.write(build_response_body(schema, cb)))[:choices][1][:message][:tool_calls][1]
    @test tc[:id] == "call_1"
    @test tc[:type] == "function"
    @test tc[:function][:name] == "bash"
    @test tc[:function][:arguments] == "{\"cmd\": \"ls\"}"

    d = only(extract_tool_deltas(schema, cb.chunks[2]))
    @test d.id === nothing && d.name === nothing && d.args == "{\"cmd\": \"ls\""
    @test only(extract_tool_deltas(schema, cb.chunks[1])).name == "bash"
end
