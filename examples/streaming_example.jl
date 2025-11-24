
using OpenRouter
using OpenRouter: HttpStreamCallback

callback = HttpStreamCallback(; out=stdout, verbose=false)

response = aigen("Count to 10 in 1 line", "anthropic:anthropic/claude-haiku-4.5"; streamcallback=callback)
#%%
callback = HttpStreamCallback(; out=stdout, verbose=false)
response = aigen("Count to 10 in 1 line", "openai:openai/gpt-5.1"; streamcallback=callback)
callback = HttpStreamCallback(; out=stdout, verbose=false)
response = aigen("Count to 100 in 1 line", "groq:moonshotai/kimi-k2-0905"; streamcallback=callback)
#%%
using OpenRouter: UserMessage, SystemMessage

callback = HttpStreamCallback(; out=stdout, verbose=false)
# response = aigen("Count to 100 in 1 line", "google-ai-studio:google/gemini-2.5-flash"; streamcallback=callback)
response = aigen([SystemMessage(content="Be conscise and helpful."), UserMessage(content="Count to 10 in 1 line")], "google-ai-studio:google/gemini-3-pro-preview"; streamcallback=callback, top_p=0.7)
#%%
aigen([UserMessage(content="Count to 10 in 1 line")], "google-ai-studio:google/gemini-3-pro-preview"; streamcallback=callback, sys_msg="Be conscise and helpful.")
#%%
callback = HttpStreamCallback(; out=stdout, verbose=true)
response = aigen("Count to 100 in 1 line", "cerebras:meta-llama/llama-3.1-8b-instruct"; streamcallback=callback)
#%%
using OpenRouter: TokenCounts
callback = HttpStreamCallback(; out=stdout, verbose=true)
# @time response = aigen("Implement me an optimal is_prime:", "google-ai-studio:google/gemini-2.5-flash", streamcallback=callback, thinkingConfig=Dict("thinkingBudget"=>2000, "include_thoughts"=>true))
# response = aigen("Count to 1-10 in 1 line:", "google-ai-studio:google/gemini-3-pro-preview", streamcallback=callback)
@time response = aigen("Count to 1-10 in 1 line:",  "google-ai-studio:google/gemini-2.5-flash", thinkingConfig=Dict("thinkingBudget"=>0, "include_thoughts"=>true), streamcallback=callback)
@time response = aigen("Count to 1-10 in 1 line:",  "google-ai-studio:google/gemini-3-pro-preview", thinkingConfig=Dict("thinkingLevel" => "low", "include_thoughts"=>true), streamcallback=callback)
#%%
using OpenRouter
using OpenRouter: HttpStreamCallback
callback = HttpStreamCallback(; out=stdout, verbose=true)
@time response = aigen("Count to 1-10 in 1 line:",  "google-ai-studio:google/gemini-3-pro-preview", streamcallback=callback,)
#%%
using OpenRouter
using OpenRouter: HttpStreamCallback
# callback = HttpStreamCallback(; out=stdout, verbose=true)
callback = HttpStreamCallback(; out=stdout)
# response = aigen("Count to 100 in 1 line", "openai:openai/gpt-5.1-codex-mini")
@time response = aigen("Count from 1 to 200 one by one in 1 line", "openai:openai/gpt-5.1-codex-mini";)
@time response = aigen("Count from 1 to 20 one by one in 1 line", "openai:openai/gpt-5.1-codex-mini"; streamcallback=callback)
;

response