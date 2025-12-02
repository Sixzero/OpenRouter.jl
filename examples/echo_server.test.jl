using OpenRouter

chat_resp = aigen("Hello world", "echo_chat:test")       # ChatCompletion
@show chat_resp
anthropic_resp = aigen("Hello world", "echo_anthropic:test")  # Anthropic
@show anthropic_resp
gemini_resp = aigen("Hello world", "echo_gemini:test")     # Gemini  
@show gemini_resp
responses_resp = aigen("Hello world", "echo_responses:test")  # Responses API
@show responses_resp