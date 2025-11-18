using OpenRouter

# Example 1: Parsed structs (default)
println("=== Parsed Models (default) ===")
models = list_models()
println("Found $(length(models)) models")

# Show first few models
for (i, model) in enumerate(models[1:min(3, length(models))])
  @show model
    println("Model $i:")
    println("  ID: $(model.id)")
    println("  Name: $(model.name)")
    println("  Context: $(model.context_length)")
    if model.pricing !== nothing
        println("  Prompt price: $(model.pricing.prompt)")
    end
    println()
end

println("="^60)

println("=== Raw JSON ===")
models_json = list_models_raw()
println(models_json[1:1000] * "...")
