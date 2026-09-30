using OpenRouter
using OpenRouter: ChatCompletionSchema, StreamChunk, HttpStreamCallback, build_response_body, AIMessage
using JSON3
using Test

# CLIProxyAPI (Codex image_generation) / OpenRouter stream generated images as
# choices[].delta.images; they must survive stream reconstruction into AIMessage.image_data.
@testset "ChatCompletion delta.images" begin
    url = "data:image/png;base64,iVBORw0KGgo="
    raws = [
        """{"id":"r","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","images":[{"type":"image_url","image_url":{"url":"$url"}}]},"finish_reason":null}]}""",
        """{"id":"r","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}""",
    ]
    cb = HttpStreamCallback()
    for r in raws
        push!(cb.chunks, StreamChunk(data=r, json=JSON3.read(r)))
    end
    schema = ChatCompletionSchema()
    body = build_response_body(schema, cb)
    result = JSON3.read(JSON3.write(body), Dict{String,Any})

    @test OpenRouter.extract_images(schema, result) == [url]
    msg = AIMessage(schema, result)
    @test msg.image_data == [url]
    @test msg.content == ""

    # No images -> nothing
    @test OpenRouter.extract_images(schema, Dict{String,Any}("choices" => [Dict("message" => Dict("content" => "hi"))])) === nothing
    # Explicit null / remote URLs are ignored (only inline data URLs are decodable)
    @test OpenRouter.extract_images(schema, Dict{String,Any}("choices" => [Dict("message" => Dict("images" => nothing))])) === nothing
    @test OpenRouter.extract_images(schema, Dict{String,Any}("choices" => [Dict("message" => Dict("images" => [Dict("image_url" => Dict("url" => "https://x/y.png"))]))])) === nothing
end
