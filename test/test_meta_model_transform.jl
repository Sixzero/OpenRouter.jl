using Test, OpenRouter

@testset "meta_model_transform" begin
    @test OpenRouter.meta_model_transform("meta/muse-spark-1.3") == "muse-spark-1.3"
    @test OpenRouter.meta_model_transform("meta/muse-wonder") == "muse-wonder-1.0"
    @test OpenRouter.meta_model_transform("muse-wonder-1.0") == "muse-wonder-1.0"
end
