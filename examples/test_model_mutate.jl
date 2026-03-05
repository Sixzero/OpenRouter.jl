#!/usr/bin/env julia
# Test core models via CLIProxyAPI local proxy.
# Run: julia examples/test_model_mutate.jl

using OpenRouter
using OpenRouterCLIProxyAPI

setup_cli_proxy!(; mutate=true)

# Also route google-ai-studio through the proxy (identity transform — proxy uses same model IDs)
# google-ai-studio models have "google/" prefix; proxy expects bare name (e.g. "gemini-3.1-pro-preview")
google_transform(id) = replace(id, r"^google/" => "")
OpenRouter.add_provider("google-ai-studio", "http://localhost:8317/v1", "Bearer", "CLIPROXYAPI_API_KEY",
    Dict{String,String}(), google_transform, OpenRouter.ChatCompletionSchema())

MODELS = [
    "anthropic:anthropic/claude-sonnet-4.6",
    "openai:openai/gpt-5.3-codex",
    "google-ai-studio:google/gemini-2.5-flash",
]

cb = HttpStreamCallback(; out=stdout, verbose=false)

for model in MODELS
    println("\n=== $model ===")
    try
        r = aigen("Say hello in one sentence.", model; streamcallback=cb)
        println()
    catch e
        println("\nERROR: $e")
    end
end
