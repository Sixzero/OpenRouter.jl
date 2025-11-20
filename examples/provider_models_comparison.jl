using OpenRouter: get_provider_info, transform_model_name



println("🤖 Provider Models Comparison")
println("="^50)

# List of providers to test
providers_to_test = [
    # "cerebras", "groq", "deepseek", "xai", "openai",
    "anthropic"
]

for provider in providers_to_test
    println("\n🔍 Testing $provider:")
    
    # Get OpenRouter models
    or_models = nothing
    try
        or_models = list_models(provider)
        println("  📡 OpenRouter: $(length(or_models)) models")
    catch e
        println("  ❌ OpenRouter failed: $e")
        continue
    end
    
    # Get native models
    native_models = nothing
    try
        native_models = list_native_models(provider)
        println("  🔗 Native API: $(length(native_models)) models")
    catch e
        println("  ❌ Native API failed: $e")
        continue
    end
    
    
    # Find matched and unmatched models after transformation
    if or_models !== nothing && native_models !== nothing
        provider_info = get_provider_info(provider)
        native_ids = Set([get(m, "id", "") for m in native_models])
        
        matched = String[]
        unmatched = String[]
        
        for or_model in or_models
            transformed_id = transform_model_name(provider_info, or_model.id)
            if transformed_id ∈ native_ids
                push!(matched, or_model.id)
            else
                push!(unmatched, or_model.id)
            end
        end
        
        # # Show native model IDs
        println("  📋 Native model IDs:")
        for model in native_models
            println("    $(get(model, "id", "N/A"))")
        end
        if !isempty(matched)
            println("  ✅ Matched OpenRouter models:")
            for model_id in matched
                transformed = transform_model_name(provider_info, model_id)
                if transformed != model_id
                    println("    $model_id → $transformed")
                else
                    println("    $model_id")
                end
            end
        end
        
        if !isempty(unmatched)
            println("  ❌ Unmatched OpenRouter models (need better transformation):")
            for model_id in unmatched
                transformed = transform_model_name(provider_info, model_id)
                if transformed != model_id
                    println("    $model_id → $transformed")
                else
                    println("    $model_id")
                end
            end
        else
            println("  ✅ All models match!")
        end
    end
    
    sleep(0.5)
end
