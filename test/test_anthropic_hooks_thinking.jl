using Test, JSON3, OpenRouter
using OpenRouter: AnthropicSchema, StreamChunk

# A thinking_delta must reach reasoning_formatter only — never content_formatter too.
@testset "HttpStreamHooks: Anthropic thinking not duplicated as text" begin
    raws = [
        """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}""",
        """{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}""",
    ]
    reasoning, text = String[], String[]
    cb = OpenRouter.HttpStreamHooks(schema=AnthropicSchema(), out=devnull,
        reasoning_formatter = t -> (push!(reasoning, t); nothing),
        content_formatter   = t -> (push!(text, t); nothing))
    for r in raws
        OpenRouter.callback(cb, StreamChunk(data=r, json=JSON3.read(r)))
    end
    @test reasoning == ["hmm"]
    @test text == ["Hi"]
end
