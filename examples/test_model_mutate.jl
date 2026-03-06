#!/usr/bin/env julia
# Test core models via CLIProxyAPI local proxy.
# Run: julia examples/test_model_mutate.jl

using OpenRouter
using OpenRouterCLIProxyAPI

setup_cli_proxy!(; mutate=true)  # inject mutate (includes google-ai-studio override)

MODELS = [
    "anthropic:anthropic/claude-sonnet-4.6",
    "openai:openai/gpt-5.3-codex",
    "openai:openai/gpt-5.3-chat",
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
