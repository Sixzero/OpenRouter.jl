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
        if isempty(get(ENV, "OPENROUTER_API_KEY", ""))
            @info "Skipping cache operations tests: OPENROUTER_API_KEY not set"
        else
            cache = update_db(fetch_endpoints = false)
            @test length(cache.models) > 0

            # Pick one model id
            any_id = first(keys(cache.models))
            cm = get_model(any_id, fetch_endpoints = false)
            @test cm !== nothing
            @test cm.model.id == any_id
        end
    end
    
    @testset "Provider model parsing" begin
        provider, model = OpenRouter.parse_provider_model("Together:moonshotai/kimi-k2-thinking")
        @test provider == "Together"
        @test model == "moonshotai/kimi-k2-thinking"
    end

    @testset "cost calculation with cache" begin
        pricing = Pricing(prompt = 0.000002, completion = 0.000004,
                          input_cache_read = 0.000001, input_cache_write = 0.0000015,
                          internal_reasoning = nothing, input_audio_cache = nothing,
                          discount = nothing)
        tokens = Dict(
            :prompt_tokens => 1000,
            :completion_tokens => 500,
            :input_cache_read => 200,
            :input_cache_write => 100,
        )
        cost = calculate_cost(pricing, tokens)
        @test isapprox(cost, 1000*0.000002 + 500*0.000004 + 200*0.000001 + 100*0.0000015; atol=1e-10)
    end

    @testset "calculate_cost with ProviderEndpoint" begin
        pricing = Pricing(
            prompt = 0.000002,
            completion = 0.000004,
            input_cache_read = 0.000001,
            input_cache_write = 0.0000015,
            internal_reasoning = nothing,
            input_audio_cache = nothing,
            discount = nothing,
        )
        endpoint = ProviderEndpoint(
            # ... existing required fields ...,
            pricing = pricing,
        )
        tokens = Dict(
            :prompt_tokens => 1000,
            :completion_tokens => 500,
            :input_cache_read => 200,
            :input_cache_write => 100,
        )
        cost = calculate_cost(endpoint, tokens)
        @test isapprox(cost, 1000*0.000002 + 500*0.000004 + 200*0.000001 + 100*0.0000015; atol=1e-10)
    end

    # Include custom provider tests
    include("test_custom_providers.jl")
end
