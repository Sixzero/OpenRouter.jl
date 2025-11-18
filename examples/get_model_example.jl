using OpenRouter
using OpenRouter: get_model, search_models

# Example 1: Get a specific model (auto-initializes cache on first use)
println("=== Getting a specific model ===")
model = get_model("moonshotai/kimi-k2-thinking")
if model !== nothing
    println("Found: $(model.model.name)")
    println("Context length: $(model.model.context_length)")
    if model.endpoints !== nothing
        println("Endpoints available: $(length(model.endpoints.endpoints))")
    end
else
    println("Model not found")
end

println("\n=== Searching for models ===")
# Example 2: Search for models
gpt_models = search_models("deepseek")
println("Found $(length(gpt_models)) models containing 'gpt':")
for cached in gpt_models[1:min(2, end)]  # Show first 5
    @show cached.endpoints
    println("  - $(cached.model.id): $(cached.model.name)")
end

println("\n=== Getting model with endpoints ===")
# Example 3: Get model with endpoints
model_with_endpoints = get_model("openai/gpt-4", fetch_endpoints=true)
if model_with_endpoints !== nothing && model_with_endpoints.endpoints !== nothing
    println("$(model_with_endpoints.model.name) has $(length(model_with_endpoints.endpoints.endpoints)) endpoints")
    for ep in model_with_endpoints.endpoints.endpoints[1:min(3, end)]
        println("  - $(ep.provider_name): $(ep.name)")
    end
end