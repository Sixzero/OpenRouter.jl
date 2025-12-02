# Precompilation workload using minimal echo server

using PrecompileTools
using HTTP, JSON3

@setup_workload begin
  
  PRECOMPILE_PORT::Int = 18787
  ECHO_PROVIDERS::Vector{String} = ["echo_chat", "echo_anthropic", "echo_gemini", "echo_responses"]
  
  # Minimal responses for each schema type
  RESPONSES::Dict{String,String} = Dict(
      "chat" => """{"id":"x","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}""",
      "messages" => """{"id":"x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}""",
      "Content" => """{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}""",
      "responses" => """{"id":"x","object":"response","status":"completed","output":[{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"""
  )
    
    handler(req) = HTTP.Response(200, ["Content-Type" => "application/json"]; 
        body = get(RESPONSES, first(k for k in keys(RESPONSES) if contains(req.target, k)), "{}"))
    
    # Swap ports for echo providers
    swap_port!(from, to) = for name in ECHO_PROVIDERS
        haskey(PROVIDER_INFO, name) || continue
        old = PROVIDER_INFO[name]
        PROVIDER_INFO[name] = ProviderInfo(replace(old.base_url, from => to), 
            old.auth_header_format, old.api_key_env_var, old.default_headers,
            old.model_name_transform, old.schema, old.notes)
    end
    
    swap_port!("8787", string(PRECOMPILE_PORT))
    
    @compile_workload begin
        server = HTTP.serve!(handler, "127.0.0.1", PRECOMPILE_PORT)
        try
            for p in ECHO_PROVIDERS
                aigen(; prompt="hi", model="$p:test")
            end
        finally
            close(server)
        end
    end
    
    swap_port!(string(PRECOMPILE_PORT), "8787")
end
