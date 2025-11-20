using OpenRouter

# Custom hooks example
callback = HttpStreamHooks(
    on_done = () -> "✓ Generation complete!",
    on_start = () -> "🚀 Starting generation...",
    on_meta_ai = (tokens, cost, elapsed) -> begin
        elapsed_str = elapsed !== nothing ? " ($(round(elapsed, digits=2))s)" : ""
        return "\nCost: \$$(round(cost, digits=6))$elapsed_str"
    end,
    content_formatter = text -> uppercase(text)  # Make all content uppercase
)
# callback = HttpStreamCallback(; out=stdout)
response = aigen("Count to 1-10 in 1 line:", "anthropic:anthropic/claude-haiku-4.5"; stream_callback=callback)
response = aigen("Count to 1-10 in 1 line:", "anthropic:anthropic/claude-haiku-4.5"; stream_callback=callback)
response = aigen("Count to 1-10 in 1 line:", "anthropic:anthropic/claude-haiku-4.5")
