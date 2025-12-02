# Minimal echo server for testing and precompilation
# Shared between src/precompile.jl and tests

using HTTP, JSON3

const ECHO_PROVIDERS = ["echo_chat", "echo_anthropic", "echo_gemini", "echo_responses"]

# Minimal valid responses for each schema type
const ECHO_RESPONSES = Dict{String,String}(
    "chat" => """{"id":"x","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}""",
    "messages" => """{"id":"x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}""",
    "Content" => """{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}""",
    "responses" => """{"id":"x","object":"response","status":"completed","output":[{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"""
)

"""Route request to appropriate response based on endpoint."""
function echo_handler(req::HTTP.Request)
    for (key, response) in ECHO_RESPONSES
        contains(req.target, key) && return HTTP.Response(200, ["Content-Type" => "application/json"]; body=response)
    end
    HTTP.Response(404)
end

"""Swap echo provider ports from `from` to `to`."""
function swap_echo_port!(from, to)
    for name in ECHO_PROVIDERS
        haskey(PROVIDER_INFO, name) || continue
        old = PROVIDER_INFO[name]
        PROVIDER_INFO[name] = ProviderInfo(replace(old.base_url, string(from) => string(to)),
            old.auth_header_format, old.api_key_env_var, old.default_headers,
            old.model_name_transform, old.schema, old.notes)
    end
end

"""Run function with echo server on given port."""
function with_echo_server(f::Function, port::Int=18787)
    swap_echo_port!(8787, port)
    server = HTTP.serve!(echo_handler, "127.0.0.1", port)
    try
        f()
    finally
        close(server)
        swap_echo_port!(port, 8787)
    end
end

