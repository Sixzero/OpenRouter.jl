using OpenRouter

# Set your Groq API key (if not already in environment)
# ENV["GROQ_API_KEY"] = "your_groq_api_key_here"

# Basic usage
response = aigen(
    "Write a haiku about Julia programming", 
    "groq:moonshotai/kimi-k2-thinking"
)
println(response)
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