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

# Minimal SSE streams per schema (for requests with "stream": true)
const ECHO_SSE = Dict{String,String}(
    "chat" => """
data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}

data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}

data: [DONE]

""",
    "messages" => """
event: message_start
data: {"type":"message_start","message":{"id":"x","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

""",
    "Content" => """
data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}

""",
    "responses" => """
data: {"type":"response.output_text.delta","delta":"ok"}

data: {"type":"response.completed","response":{"id":"x","object":"response","status":"completed","output":[{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}

""",
)

_wants_stream(req::HTTP.Request) = contains(String(copy(req.body)), "\"stream\":true") ||
                                   contains(String(copy(req.body)), "\"stream\": true") ||
                                   contains(req.target, "streamGenerateContent")

"""Route request to appropriate response based on endpoint."""
function echo_handler(req::HTTP.Request)
    for (key, response) in ECHO_RESPONSES
        if contains(req.target, key)
            _wants_stream(req) &&
                return HTTP.Response(200, ["Content-Type" => "text/event-stream"]; body=ECHO_SSE[key])
            return HTTP.Response(200, ["Content-Type" => "application/json"]; body=response)
        end
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
    # Bind BEFORE mutating provider URLs so a failed bind (port in use — e.g. two
    # processes warming up simultaneously) leaves PROVIDER_INFO untouched.
    server = HTTP.serve!(echo_handler, "127.0.0.1", port)
    swap_echo_port!(8787, port)
    try
        f()
    finally
        close(server)
        swap_echo_port!(port, 8787)
    end
end

