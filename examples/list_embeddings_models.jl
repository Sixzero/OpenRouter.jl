using OpenRouter
using OpenRouter: list_embeddings_models

# Example 1: List all embedding models
println("=== Embedding Models ===")
embedding_models = list_embeddings_models()
println("Found $(length(embedding_models)) embedding models")

# Show first few models
for (i, model) in enumerate(embedding_models[1:min(30, end)])
    println("Model $i:")
    println("  ID: $(model.id)")
    println("  Name: $(model.name)")
    println("  Context: $(model.context_length)")
    if model.pricing !== nothing
        println("  Prompt price: $(model.pricing.prompt)")
    end
    if model.top_provider !== nothing
        println("  Top provider context: $(model.top_provider.context_length)")
        println("  Moderated: $(model.top_provider.is_moderated)")
    end
    println()
end

println("="^60)
