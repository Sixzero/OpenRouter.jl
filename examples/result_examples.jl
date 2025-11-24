long_text = "just a long text, we shouldn't even bother with ourself with this."^50
@show length(long_text)
response = aigen("$long_text Count to 1-10 in 1 line:", "anthropic:anthropic/claude-haiku-4.5")
response = aigen("$long_text Count to 1-10 in 1 line:", "openai:openai/gpt-5.1")
response = aigen("$long_text Count to 1-10 in 1 line:", "google-ai-studio:google/gemini-2.5-flash")
response = aigen("$long_text Count to 1-10 in 1 line:", "groq:moonshotai/kimi-k2-0905")
response = aigen("$long_text Count to 1-10 in 1 line:", "cerebras:openai/gpt-oss-120b")
#%%
response = aigen("Count to 1-10 in 1 linee:", "gemf")
#%%
response = aigen("Count to 1-10 in 1 linee:", "gemf", streamcallback=HttpStreamCallback())
#%%
@show response.tokens