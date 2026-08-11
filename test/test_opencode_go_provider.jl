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

    parsed_info, model_id, endpoint = parse_provider_model("opencode_go:moonshotai/kimi-k3")
    @test parsed_info === info
    @test model_id == "kimi-k3"
    @test endpoint.provider_name == "opencode_go"
    @test endpoint.pricing !== nothing

    payload = OpenRouter.build_payload(ChatCompletionSchema(), "hi", "kimi-k3", nothing; top_p=0.9)
    @test payload["top_p"] == 0.95
end

@testset "OpenCode Go export" begin
    @test OPENCODE_GO_EXCLUDED_MODELS == Set(["gpt-5.6-luna", "grok-4.5"])

    catalog = Any[
        Dict("id" => "moonshotai/kimi-k3", "name" => "MoonshotAI: Kimi K3", "endpoints" => Any[Dict(
            "provider_name" => "Moonshot-AI", "context_length" => 1_000_000,
            "max_completion_tokens" => 65_536,
            "pricing" => Dict("prompt" => 3e-6, "completion" => 15e-6),
        )]),
    ]

    if !isempty(get(ENV, "OPENCODE_API_KEY", ""))
        specs = build_opencode_go_specs("opencode_go", catalog)
        ids = Set(spec["id"] for spec in specs)
        @test "moonshotai/kimi-k3" in ids
        @test "openai/gpt-5.6-luna" ∉ ids
        @test "x-ai/grok-4.5" ∉ ids
        kimi = only(filter(spec -> spec["id"] == "moonshotai/kimi-k3", specs))
        @test kimi["endpoints"][1]["pricing"] == catalog[1]["endpoints"][1]["pricing"]
        @test all(spec["endpoints"][1]["provider_name"] == "opencode_go" for spec in specs)
    else
        @info "Skipping OpenCode Go live export test (set OPENCODE_API_KEY to enable)"
    end
end
