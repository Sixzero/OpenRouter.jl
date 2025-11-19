using OpenRouter
using OpenRouter: update_db
using Test
using Aqua

@testset "OpenRouter.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        # Aqua.test_all(OpenRouter)
    end
    
    @testset "Basic functionality" begin
        @test isa(OpenRouter.list_models, Function)
    end
    
    @testset "Cache operations" begin
        cache = update_db(fetch_endpoints = false)
        @test length(cache.models) > 0

        # Pick one model id
        any_id = first(keys(cache.models))
        cm = get_model(any_id, fetch_endpoints = false)
        @test cm !== nothing
        @test cm.model.id == any_id
    end
    
    @testset "Provider model parsing" begin
        provider, model = OpenRouter.parse_provider_model("Together:moonshotai/kimi-k2-thinking")
        @test provider == "Together"
        @test model == "moonshotai/kimi-k2-thinking"
    end
    
    # Include custom provider tests
    include("test_custom_providers.jl")
end
