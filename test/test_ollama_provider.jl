using Test
using OpenRouter
using OpenRouter: get_provider_info, extract_provider_from_model

@testset "Ollama Provider Basic" begin
    info = get_provider_info("ollama")
    @test info !== nothing
    @test info.base_url == "http://localhost:11434/v1"
    @test info.schema isa OpenRouter.ChatCompletionSchema
    @test info.api_key_env_var === nothing

    # Model parsing with double colon
    provider = extract_provider_from_model("ollama:smollm:360m")
    @test provider == "ollama"
end