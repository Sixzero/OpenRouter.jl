using OpenRouter
using OpenRouter: ChatCompletionSchema, StreamChunk, HttpStreamCallback, build_response_body, AIMessage,
    split_content_blocks, extract_content, extract_reasoning
using OpenRouter: extract_reasoning_from_chunk
using JSON3
using Test

# Mistral reasoning models (mistral-large-4) send `content` as a block array of
# thinking/text parts, both in full responses and in stream deltas.
@testset "Mistral content blocks" begin
    schema = ChatCompletionSchema()

    @testset "split_content_blocks" begin
        @test split_content_blocks("plain") == ("plain", nothing)
        @test split_content_blocks(nothing) == (nothing, nothing)
        blocks = JSON3.read("""[{"type":"thinking","thinking":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"closed":true},
                                {"type":"text","text":"x"},{"type":"text","text":"y"}]""")
        @test split_content_blocks(blocks) == ("xy", "ab")
        @test split_content_blocks(JSON3.read("""[{"type":"thinking","thinking":"raw"}]""")) == (nothing, "raw")
        # single-block delta: the chunk's own string passes through untouched
        single = JSON3.read("""[{"type":"text","text":"solo"}]""")
        @test split_content_blocks(single) == ("solo", nothing)
    end

    @testset "full response" begin
        raw = """{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","tool_calls":null,
            "content":[{"type":"thinking","thinking":[{"type":"text","text":"think"}],"closed":true},{"type":"text","text":"Hi there friend"}]}}],
            "usage":{"prompt_tokens":9,"completion_tokens":10,"total_tokens":19}}"""
        result = JSON3.read(raw, Dict{String,Any})
        @test extract_content(schema, result) == "Hi there friend"
        @test extract_reasoning(schema, result) == "think"
        msg = AIMessage(schema, result)
        @test msg.content == "Hi there friend"
        @test msg.reasoning == "think"
    end

    @testset "stream" begin
        raws = [
            """{"id":"s","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}""",
            """{"id":"s","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"th"}],"closed":true}]},"finish_reason":null}]}""",
            """{"id":"s","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"ink"}],"closed":true},{"type":"text","text":"Hi"}]},"finish_reason":null}]}""",
            """{"id":"s","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":5,"total_tokens":14}}""",
        ]
        chunks = [StreamChunk(data=r, json=JSON3.read(r)) for r in raws]
        @test extract_reasoning_from_chunk(schema, chunks[2]) == "th"
        @test extract_content(schema, chunks[2]) === nothing
        @test extract_reasoning_from_chunk(schema, chunks[3]) == "ink"
        @test extract_content(schema, chunks[3]) == "Hi"
        @test extract_content(schema, chunks[4]) == " there"

        cb = HttpStreamCallback(out=devnull)
        append!(cb.chunks, chunks)
        body = build_response_body(schema, cb)
        msg = AIMessage(schema, JSON3.read(JSON3.write(body), Dict{String,Any}))
        @test msg.content == "Hi there"
        @test msg.reasoning == "think"

        # live printing: a chunk with both parts prints both
        io = IOBuffer()
        cb2 = OpenRouter.HttpStreamHooks(schema=schema, out=io)
        for c in chunks
            OpenRouter.callback(cb2, c)
        end
        s = replace(String(take!(io)), r"\e\[\d+m" => "")   # strip ANSI colors
        @test occursin("think", s) && occursin("Hi there", s)
        @test findfirst("think", s).stop < findfirst("Hi", s).start
    end
end
