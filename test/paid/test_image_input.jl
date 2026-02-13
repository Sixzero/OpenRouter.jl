using OpenRouter
using OpenRouter: UserMessage, AIMessage
using Test
using Base64

@testset "Image Input" begin
    # 200x60 PNG with "banana" written on it
    img_b64 = base64encode(read(joinpath(@__DIR__, "..", "assets", "banana.png")))
    msg = UserMessage(content="", image_data=["data:image/png;base64,$img_b64"])

    resp = aigen(msg, "anthropic:anthropic/claude-haiku-4.5"; sys_msg="Reply with only the word shown in the image.")
    @test resp isa AIMessage
    @test !isempty(resp.content)
    @test resp.tokens !== nothing
    println("✓ Image input: $(resp.content)")
end
