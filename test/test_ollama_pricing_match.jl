using Test
using JSON3
using OpenRouter

# Load the export script's matcher without running main() (guarded by PROGRAM_FILE).
include(joinpath(@__DIR__, "..", "scripts", "export_models_json.jl"))

@testset "Ollama catalog price matching" begin
    # Minimal synthetic catalog id list covering the real-world cases.
    catalog = [
        "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        "deepseek/deepseek-v3.2", "deepseek/deepseek-v3.1-terminus",
        "z-ai/glm-5", "z-ai/glm-5.1", "z-ai/glm-5.2",
        "moonshotai/kimi-k2", "moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6",
        "qwen/qwen3-coder", "qwen/qwen3-coder-next", "qwen/qwen3.5-9b", "qwen/qwen3.5-397b-a17b",
        "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it",
        "nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-ultra-550b-a55b:free",   # free must be skipped
        "minimax/minimax-m2", "minimax/minimax-m2.7",
        "mistralai/ministral-3b-2512", "mistralai/ministral-8b-2512", "mistralai/ministral-14b-2512",
    ]

    # Version-sensitive: must pick the exact version, not the base.
    @test find_catalog_match("gpt-oss:20b", catalog)   == "openai/gpt-oss-20b"
    @test find_catalog_match("gpt-oss:120b", catalog)  == "openai/gpt-oss-120b"
    @test find_catalog_match("deepseek-v3.2", catalog) == "deepseek/deepseek-v3.2"
    @test find_catalog_match("glm-5.1", catalog)       == "z-ai/glm-5.1"
    @test find_catalog_match("glm-5.2", catalog)       == "z-ai/glm-5.2"
    @test find_catalog_match("glm-5", catalog)         == "z-ai/glm-5"
    @test find_catalog_match("kimi-k2.6", catalog)     == "moonshotai/kimi-k2.6"
    @test find_catalog_match("minimax-m2.7", catalog)  == "minimax/minimax-m2.7"

    # Size-sensitive: gemma sizes must not collapse to the smallest.
    @test find_catalog_match("gemma3:27b", catalog) == "google/gemma-3-27b-it"
    @test find_catalog_match("gemma3:12b", catalog) == "google/gemma-3-12b-it"
    @test find_catalog_match("qwen3.5:397b", catalog) == "qwen/qwen3.5-397b-a17b"

    # Ministral: Ollama's "3" is a generation tag, ":14b/:8b/:3b" the size. Must
    # map to the same-SIZE catalog twin, not collapse to the 3B (cheapest) one.
    @test find_catalog_match("ministral-3:14b", catalog) == "mistralai/ministral-14b-2512"
    @test find_catalog_match("ministral-3:8b", catalog)  == "mistralai/ministral-8b-2512"
    @test find_catalog_match("ministral-3:3b", catalog)  == "mistralai/ministral-3b-2512"

    # MoE "<total>b-a<active>b" names: parameter size must read the TOTAL count.
    @test _param_size("qwen3.5-397b-a17b") == 397
    @test _param_size("nemotron-3-ultra-550b-a55b") == 550
    @test _param_size("nemotron-3-nano-30b-a3b") == 30
    @test _param_size("minimax-m2") === nothing

    # Free catalog variants must never be chosen (they're $0).
    @test find_catalog_match("nemotron-3-ultra", catalog) == "nvidia/nemotron-3-ultra-550b-a55b"

    # No family in catalog -> no match -> the model is dropped.
    @test find_catalog_match("rnj-1:8b", catalog) === nothing
    @test find_catalog_match("devstral-small-2:24b", catalog) === nothing

    # Guard against silent drops: check the RAW Ollama Cloud list (live, gated on
    # OLLAMA_API_KEY) against the current OpenRouter catalog. The exported JSON
    # can't be used here — dropped models are already gone from it. If a new
    # Ollama model has no catalog twin, this flags it (export would drop it).
    if !isempty(get(ENV, "OLLAMA_API_KEY", ""))
        ollama_ids = [String(get(m, "model", get(m, "name", "")))
                      for m in OpenRouter.list_native_models("ollama_cloud")]
        filter!(!isempty, ollama_ids)
        catalog_ids = [String(m.id) for m in OpenRouter.list_models()]

        unmatched = [oid for oid in ollama_ids if find_catalog_match(oid, catalog_ids) === nothing]
        # Models with genuinely no catalog twin; update if Ollama's list changes.
        known_unmatched = Set(["rnj-1:8b", "devstral-small-2:24b"])
        surprises = setdiff(Set(unmatched), known_unmatched)
        isempty(surprises) || @warn "New unmatched Ollama models (will be DROPPED from export)" surprises
        @test isempty(surprises)
    else
        @info "Skipping Ollama->catalog guard (set OLLAMA_API_KEY to enable)"
    end
end
