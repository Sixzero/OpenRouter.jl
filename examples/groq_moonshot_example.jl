using OpenRouter

# Set your Groq API key (if not already in environment)
# ENV["GROQ_API_KEY"] = "your_groq_api_key_here"

# Basic usage
@time response = aigen(
    "Write a haiku about Julia programming", 
    "groq:moonshotai/kimi-k2-0905"
    # "anthropic:anthropic/claude-haiku-4.5"
    # "openai:openai/gpt-5.1"
)
println(response)
#%%
using OpenRouter

# list_models("openai")
mm = list_models("openai")
mm = list_models("google")
mm = list_models("anthropic")
mm = list_models("groq")
mm = list_models("cerebras")
display([m.id for m in mm])

#%%
get_model("moonshotai/kimi-k2-0905:exacto").endpoints.endpoints .|> println
#%%
using OpenRouter: get_global_cache
models = list_models()
[m.id for m in models if contains(m.id, "kimi-k2")] .|> println
#%%
# Show endpoints for groq
# groq_eps = list_provider_endpoints("anthropic")
groq_eps = list_provider_endpoints("groq")

for m in groq_eps
    println(m)
end
# println("Groq hosts $(length(groq_eps)) endpoints")
#%%

# With parameters
response = aigen(
    "Explain quantum computing in simple terms", 
    "groq:moonshotai/kimi-k2-thinking";
    temperature=0.7,
    max_tokens=300
)
println(response)

# Raw response with usage info
raw_response = aigen_raw(
    "What are the benefits of Julia?", 
    "groq:moonshotai/kimi-k2-thinking"
)
println("Tokens used: ", raw_response.usage.total_tokens)