using Test
using OpenRouter
using OpenRouter: get_provider_info, parse_provider_model, ChatCompletionSchema

include(joinpath(@__DIR__, "..", "scripts", "export_models_json.jl"))

@testset "OpenCode Go provider" begin
    info = get_provider_info("opencode_go")
    @test info !== nothing
    @test info.base_url == "https://opencode.ai/zen/go/v1"
    @test info.api_key_env_var == "OPENCODE_API_KEY"
    @test info.schema isa ChatCompletionSchema

    parsed_info, model_id, endpoint = parse_provider_model("opencode_go:kimi-k3")
    @test parsed_info === info
    @test model_id == "kimi-k3"
    @test endpoint.provider_name == "opencode_go"
end

@testset "OpenCode Go export" begin
    @test OPENCODE_GO_EXCLUDED_MODELS == Set(["gpt-5.6-luna", "grok-4.5"])

    catalog = Any[Dict("id" => "kimi-k3", "endpoints" => Any[Dict("provider_name" => "Moonshot-AI")])]
    native = Any[
        Dict("id" => "kimi-k3", "endpoints" => Any[Dict("provider_name" => "opencode_go")]),
        Dict("id" => "glm-5", "endpoints" => Any[Dict("provider_name" => "opencode_go")]),
    ]
    merge_model_specs!(catalog, native)
    @test length(catalog) == 2
    @test [ep["provider_name"] for ep in catalog[1]["endpoints"]] == ["Moonshot-AI", "opencode_go"]

    if !isempty(get(ENV, "OPENCODE_API_KEY", ""))
        specs = build_opencode_go_specs()
        ids = Set(spec["id"] for spec in specs)
        @test "kimi-k3" in ids
        @test "gpt-5.6-luna" ∉ ids
        @test "grok-4.5" ∉ ids
        @test all(spec["endpoints"][1]["provider_name"] == "opencode_go" for spec in specs)
    else
        @info "Skipping OpenCode Go live export test (set OPENCODE_API_KEY to enable)"
    end
end
