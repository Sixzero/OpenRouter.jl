using OpenRouter

# Update the database with all models and endpoints
println("Starting database update...")

# First run: fetch all models and endpoints
cache = update_db(
    fetch_endpoints=true,
    max_concurrent=5  # Be gentle with the API
)

println("\nDatabase updated successfully!")
println("Total models cached: $(length(cache.models))")

# Show some statistics
models_with_endpoints = sum(cached.endpoints !== nothing for cached in values(cache.models))
println("Models with endpoints: $models_with_endpoints")

# Example: Get a specific model with endpoints
if haskey(cache.models, "moonshotai/kimi-k2-thinking")
    cached_model = cache.models["moonshotai/kimi-k2-thinking"]
    if cached_model.endpoints !== nothing
        println("\nExample - Kimi K2 Thinking:")
        println("  Model: $(cached_model.model.name)")
        println("  Endpoints: $(length(cached_model.endpoints.endpoints))")
        println("  First endpoint provider: $(cached_model.endpoints.endpoints[1].provider_name)")
    end
end

# Example: Incremental update (only fetch new models)
println("\nRunning incremental update...")
cache2 = update_db(
    full_refresh=false,
    fetch_endpoints=true
)