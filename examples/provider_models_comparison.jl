using OpenRouter: list_models, list_native_models, add_provider, ChatCompletionSchema
using OpenRouter: get_provider_info, transform_model_name

# Wrapper: accepts String or Pair{String, Vector{String}}
# - String: provider (e.g. "anthropic") -> returns (provider, model) pairs
# - Pair: native_provider => [openrouter_provider1, ...] -> concats those OpenRouter lists
list_models_wrapped(p::String) = [(p, m) for m in list_models(p)]
list_models_wrapped(p::Pair) = vcat([[(or_p, m) for m in list_models(or_p)] for or_p in p.second]...)

# Extract native provider name
native_provider(p::String) = p
native_provider(p::Pair) = p.first

println("🤖 Provider Models Comparison for transformation function")
println("="^50)

# Setup CLI Proxy API provider
add_provider("cli_proxy_api", "http://localhost:8317/v1", "Bearer", "CLIPROXYAPI_API_KEY",
    Dict{String,String}(), nothing, ChatCompletionSchema())

# List of providers to test
providers_to_test = [
    # "cerebras", "groq", "deepseek", "xai",
    # "anthropic", "siliconflow",
    # "deepseek",
    # "openai",
    "cli_proxy_api" => ["anthropic", "openai"],
]

for provider in providers_to_test
    native_p = native_provider(provider)
    @show provider
    println("\n🔍 Testing $provider:")
    
    # Get OpenRouter models (concatenated if Pair)
    or_models = list_models_wrapped(provider)
    println("  📡 OpenRouter: $(length(or_models)) models")
    
    # Get native models
    native_models = list_native_models(native_p)
    println("  🔗 Native API: $(length(native_models)) models")
    
    
    # Find matched and unmatched models after transformation
    if or_models !== nothing && native_models !== nothing
        native_ids = Set([get(m, "id", "") for m in native_models])
        
        matched = Tuple{String,String}[]   # (or_provider, model_id)
        unmatched = Tuple{String,String}[]

        for (or_provider, or_model) in or_models
            pinfo = get_provider_info(or_provider)
            transformed_id = pinfo !== nothing ? transform_model_name(pinfo, or_model.id) : or_model.id
            if transformed_id ∈ native_ids
                push!(matched, (or_provider, or_model.id))
            else
                push!(unmatched, (or_provider, or_model.id))
            end
        end
        
        # Show native model IDs
        println("  📋 Native model IDs:")
        for model in native_models
            println("    $(get(model, "id", "N/A"))")
        end
        if !isempty(matched)
            println("  ✅ Matched OpenRouter models:")
            for (or_provider, model_id) in matched
                pinfo = get_provider_info(or_provider)
                transformed = pinfo !== nothing ? transform_model_name(pinfo, model_id) : model_id
                println("    $model_id → $transformed")
            end
        end
        
        if !isempty(unmatched)
            println("  ❌ Unmatched OpenRouter models:")
            for (or_provider, model_id) in unmatched
                pinfo = get_provider_info(or_provider)
                transformed = pinfo !== nothing ? transform_model_name(pinfo, model_id) : model_id
                println("    $model_id → $transformed")
            end
        else
            println("  ✅ All models match!")
        end
    end
    
    sleep(0.5)
end
