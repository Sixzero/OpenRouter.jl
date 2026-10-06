using Test, OpenRouter, Dates, JSON3

@testset "Targeted endpoint refresh" begin
    id = "qwen/qwen3.8-27b"
    model = OpenRouter.OpenRouterModel(id, "Qwen 3.8", "Test", 262144, nothing, nothing, nothing)
    endpoints(provider) = OpenRouter.parse_endpoints(JSON3.write(Dict("data" => Dict(
        "id" => id, "name" => "Qwen 3.8", "endpoints" => [Dict(
            "name" => provider, "model_name" => "Qwen 3.8", "provider_name" => provider,
            "tag" => lowercase(provider) * "/bf16", "pricing" => Dict("prompt" => "0.00000015", "completion" => "0.000001875"))]))))
    stale = OpenRouter.CachedModel(model, endpoints("Chutes"), now(), true)
    original = OpenRouter.GLOBAL_CACHE[]
    file = OpenRouter.get_cache_file()
    original_file = isfile(file) ? read(file) : nothing
    calls = Ref(0)
    fetcher = (model_id, key) -> begin
        @test model_id == id
        calls[] += 1
        endpoints("DeepInfra")
    end
    try
        OpenRouter.GLOBAL_CACHE[] = OpenRouter.ModelCache(Dict(id => stale), now())
        info, native, endpoint = OpenRouter.parse_provider_model("deepinfra:" * id; endpoint_fetcher=fetcher)
        @test native == "Qwen/Qwen3.8-27B"
        @test info.base_url == "https://api.deepinfra.com/v1/openai"
        @test endpoint.provider_name == "DeepInfra"
        @test calls[] == 1
        OpenRouter.parse_provider_model("deepinfra:" * id; endpoint_fetcher=fetcher)
        @test calls[] == 1  # Matching routes keep the cached fast path.

        OpenRouter.GLOBAL_CACHE[].models[id] = stale
        @test_throws ArgumentError OpenRouter.parse_provider_model("deepinfra:" * id;
            endpoint_fetcher=(id, key) -> (calls[] += 1; endpoints("Chutes")))
        @test calls[] == 2  # Only one refresh before rejecting a missing provider.

        failed = OpenRouter.CachedModel(model, nothing, now(), true)
        OpenRouter.GLOBAL_CACHE[].models[id] = failed
        @test OpenRouter.get_model(id; fetch_endpoints=true,
            endpoint_fetcher=(id, key) -> error("temporary fetch failure")) === failed
        @test OpenRouter.GLOBAL_CACHE[].models[id] === failed
        retried = OpenRouter.get_model(id; fetch_endpoints=true, endpoint_fetcher=fetcher)
        @test retried.endpoints_fetched
        @test retried.endpoints.endpoints[1].provider_name == "DeepInfra"
        @test calls[] == 3

        OpenRouter.GLOBAL_CACHE[].models[id] = stale
        @test_throws ErrorException OpenRouter.get_model(id; refresh_endpoints=true,
            endpoint_fetcher=(id, key) -> error("temporary fetch failure"))
        @test OpenRouter.GLOBAL_CACHE[].models[id] === stale
    finally
        OpenRouter.GLOBAL_CACHE[] = original
        original_file === nothing ? rm(file; force=true) : write(file, original_file)
    end
end
