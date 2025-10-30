using OpenRouter
using Test
using Aqua

@testset "OpenRouter.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(OpenRouter)
    end
    # Write your tests here.
end
