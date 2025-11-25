using OpenRouter

the_printer(tokens, cost, elapsed) = begin
        elapsed_str = elapsed !== nothing ? " ($(round(elapsed, digits=2))s)" : ""
        return "\n$tokens Cost: \$$(round(cost, digits=6))$elapsed_str"
    end
# Custom hooks example with on_done printing
callback = HttpStreamHooks(
    on_done = () -> "\n✓ Generation complete!",  # Returns string to be printed
    on_start = () -> "🚀 Starting generation...",
    on_meta_usr = the_printer,
    on_meta_ai = the_printer,
    content_formatter = text -> uppercase(text)  # Make all content uppercase
)

# Anthropic example
# response = aigen("Count to 1-5 in 1 line:", "groq:moonshotai/kimi-k2-0905"; streamcallback=callback)
response = aigen("Count to 1-5 in 1 line:", "anthropic:anthropic/claude-haiku-4.5"; streamcallback=callback)
@show response.tokens
# GPT‑5.1 example (OpenAI via OpenRouter)
# response = aigen("Count to 1-5 in 1 line:", "google-ai-studio:google/gemini-2.5-flash"; streamcallback=callback)
# response = aigen("Count to 1-5 in 1 line:", "openai:openai/gpt-5-mini"; streamcallback=callback)
#%%

#%%
# Example with caching - on_done will also print when complete
response = aigen("Count to 1-100 in 1 line:", "anthropic:anthropic/claude-haiku-4.5", streamcallback=callback, cache=:all)
