using Test
using OpenRouter
using OpenRouter: with_echo_server, ECHO_PROVIDERS, UserMessage

@testset "Echo Server" begin
    with_echo_server() do
        @testset "Schema: $provider" for provider in ECHO_PROVIDERS
            resp = aigen(; prompt="test", model="$provider:test")
            @test resp.content == "ok"
            @test resp.tokens !== nothing
        end

        @testset "Image input: $provider" for provider in ECHO_PROVIDERS
            img = UserMessage(content="", image_data=["data:image/png;base64,iVBORw0KGgo="])
            resp = aigen(img, "$provider:test")
            @test resp.content == "ok"
        end
    end
end

