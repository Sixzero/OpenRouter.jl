using Test
using OpenRouter
using OpenRouter: with_echo_server, ECHO_PROVIDERS

@testset "Echo Server" begin
    with_echo_server() do
        @testset "Schema: $provider" for provider in ECHO_PROVIDERS
            resp = aigen(; prompt="test", model="$provider:test")
            @test resp.content == "ok"
            @test resp.tokens !== nothing
        end
    end
end

