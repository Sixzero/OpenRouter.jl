using OpenRouter

# Ollama must be running: ollama serve
# And model pulled: ollama pull smollm:360m

resp = aigen("Hello from Julia!", "ollama:smollm:360m")
println(resp.content)

# Streaming also works, since we're just using ChatCompletionSchema
cb = HttpStreamCallback(; out=stdout)
resp = aigen("Count to 5", "ollama:smollm:360m"; streamcallback=cb)
println()
println("Tokens: ", resp.tokens)