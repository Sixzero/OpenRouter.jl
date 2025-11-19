
using OpenRouter
using OpenRouter: HttpStreamCallback

callback = HttpStreamCallback(; out=stdout, verbose=false)

response = aigen("Count to 10 in 1 line", "anthropic:anthropic/claude-haiku-4-5"; stream_callback=callback)
callback = HttpStreamCallback(; out=stdout, verbose=false)
response = aigen("Count to 10 in 1 line", "openai:openai/gpt-5.1"; stream_callback=callback)
callback = HttpStreamCallback(; out=stdout, verbose=false)
response = aigen("Count to 100 in 1 line", "google-ai-studio:google/gemini-2.5-flash"; stream_callback=callback)
# response = aigen("Explain how AI works in 20 lines each with 5 words", "google-ai-studio:gemini-2.5-flash"; stream_callback=callback)