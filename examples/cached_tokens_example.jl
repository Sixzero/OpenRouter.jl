# Anthropic Prompt Caching Example
# Reduces costs by caching system prompts and frequently reused context

using OpenRouter
using OpenRouter: UserMessage, SystemMessage

# Long system prompt benefits most from caching
long_system = """You are an expert Julia programmer with deep knowledge of:
- Performance optimization and type stability
- Multiple dispatch and metaprogramming  
- Package development best practices
- The scientific computing ecosystem (LinearAlgebra, DifferentialEquations, etc.)""" * ("- Here I have a long sentence, just to test the cache. 300 times"^300) * "Always provide benchmarks and type-stable code examples."

model = "anthropic:anthropic/claude-haiku-4.5"

# Cache modes:
# :system        - cache only system prompt
# :last          - cache last user message
# :all           - cache system + last 2 user messages
# :all_but_last  - cache system + second-to-last user message

# First call - cache write (slightly more expensive)
r1 = aigen([
    SystemMessage(content=long_system),
    UserMessage(content="What's the fastest way to sum a vector?")
], model; cache=:all, max_tokens=300)

println("First call tokens: ", r1.tokens)
println("Cache write: ", r1.tokens.input_cache_write)
println("Cache read:  ", r1.tokens.input_cache_read)

#%% Second call - cache hit (cheaper!)
r2 = aigen([
    SystemMessage(content=long_system),  # same system = cache hit
    UserMessage(content="How about matrix multiplication?")
], model; cache=:all_but_last, max_tokens=500)

println("\nSecond call tokens: ", r2.tokens)
println("Cache write: ", r2.tokens.input_cache_write)
println("Cache read:  ", r2.tokens.input_cache_read)  # should be > 0 now

#%% Cost comparison
println("\nCost comparison:")
println("First call:  \$", round(r1.cost, digits=6))
println("Second call: \$", round(r2.cost, digits=6))

