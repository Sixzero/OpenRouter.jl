# Multi-schema echo server for testing all API formats
# Run with: julia examples/echo_server.jl

using HTTP
using JSON3
using UUIDs

const PORT = 8787

gen_id(prefix="") = prefix * replace(string(uuid4()), "-" => "")[1:24]

# Extract text from content (handles string or array of content blocks)
function extract_text_from_content(content)
    content isa String && return content
    if content isa Vector
        for block in content
            # Anthropic/ResponseSchema content blocks: {type: "text", text: "..."} or {type: "input_text", text: "..."}
            if get(block, :type, nothing) in ("text", "input_text") && haskey(block, :text)
                return block[:text]
            end
        end
    end
    return nothing
end

# Extract user input from various request formats
function extract_input(body)
    # ResponseSchema: input field (string or array of items)
    if haskey(body, :input)
        input = body[:input]
        input isa String && return input
        if input isa AbstractVector
            # Find user message items (ResponseSchema uses type="message", role="user")
            for item in Iterators.reverse(input)
                item_type = get(item, :type, nothing)
                role = get(item, :role, nothing)
                if role == "user" || (item_type == "message" && role == "user")
                    content = get(item, :content, nothing)
                    # ResponseSchema sends content as plain string
                    content isa String && return content
                    # Or as content blocks array
                    text = extract_text_from_content(content)
                    !isnothing(text) && return text
                end
            end
        end
    end
    # ChatCompletion/Anthropic: messages field
    if haskey(body, :messages)
        for msg in reverse(collect(body[:messages]))
            if get(msg, :role, nothing) == "user"
                text = extract_text_from_content(get(msg, :content, nothing))
                !isnothing(text) && return text
            end
        end
    end
    # Gemini: contents field
    if haskey(body, :contents)
        for item in reverse(collect(body[:contents]))
            if get(item, :role, nothing) == "user"
                parts = get(item, :parts, [])
                !isempty(parts) && return get(first(parts), :text, "[no text]")
            end
        end
    end
    return "[no input]"
end

# ─────────────────────────────────────────────────────────────────────────────
# Response builders for each schema
# ─────────────────────────────────────────────────────────────────────────────

function build_chat_completion_response(text, model, stream)
    id = "chatcmpl-" * gen_id()
    tokens_in, tokens_out = length(split(text)) + 5, length(split(text))
    
    if stream
        events = String[]
        # Stream chunks
        for (i, char) in enumerate(text)
            chunk = Dict("id" => id, "object" => "chat.completion.chunk", "model" => model,
                "choices" => [Dict("index" => 0, "delta" => Dict("content" => string(char)), "finish_reason" => nothing)])
            push!(events, "data: $(JSON3.write(chunk))\n\n")
        end
        # Final chunk with finish_reason
        push!(events, "data: $(JSON3.write(Dict("id" => id, "object" => "chat.completion.chunk", "model" => model,
            "choices" => [Dict("index" => 0, "delta" => Dict(), "finish_reason" => "stop")])))\n\n")
        push!(events, "data: [DONE]\n\n")
        return join(events)
    else
        return JSON3.write(Dict(
            "id" => id, "object" => "chat.completion", "created" => round(Int, time()), "model" => model,
            "choices" => [Dict("index" => 0, "message" => Dict("role" => "assistant", "content" => text), "finish_reason" => "stop")],
            "usage" => Dict("prompt_tokens" => tokens_in, "completion_tokens" => tokens_out, "total_tokens" => tokens_in + tokens_out)
        ))
    end
end

function build_anthropic_response(text, model, stream)
    id = "msg_" * gen_id()
    tokens_in, tokens_out = length(split(text)) + 5, length(split(text))
    
    if stream
        events = String[]
        push!(events, "event: message_start\ndata: $(JSON3.write(Dict("type" => "message_start", 
            "message" => Dict("id" => id, "type" => "message", "role" => "assistant", "model" => model, 
                "content" => [], "usage" => Dict("input_tokens" => tokens_in, "output_tokens" => 0)))))\n\n")
        push!(events, "event: content_block_start\ndata: $(JSON3.write(Dict("type" => "content_block_start", 
            "index" => 0, "content_block" => Dict("type" => "text", "text" => ""))))\n\n")
        for char in text
            push!(events, "event: content_block_delta\ndata: $(JSON3.write(Dict("type" => "content_block_delta", 
                "index" => 0, "delta" => Dict("type" => "text_delta", "text" => string(char)))))\n\n")
        end
        push!(events, "event: content_block_stop\ndata: $(JSON3.write(Dict("type" => "content_block_stop", "index" => 0)))\n\n")
        push!(events, "event: message_delta\ndata: $(JSON3.write(Dict("type" => "message_delta", 
            "delta" => Dict("stop_reason" => "end_turn"), "usage" => Dict("output_tokens" => tokens_out))))\n\n")
        push!(events, "event: message_stop\ndata: $(JSON3.write(Dict("type" => "message_stop")))\n\n")
        return join(events)
    else
        return JSON3.write(Dict(
            "id" => id, "type" => "message", "role" => "assistant", "model" => model,
            "content" => [Dict("type" => "text", "text" => text)],
            "stop_reason" => "end_turn",
            "usage" => Dict("input_tokens" => tokens_in, "output_tokens" => tokens_out)
        ))
    end
end

function build_gemini_response(text, model, stream)
    tokens_in, tokens_out = length(split(text)) + 5, length(split(text))
    
    if stream
        events = String[]
        # Gemini streams full candidates with accumulated text
        accumulated = ""
        for char in text
            accumulated *= char
            chunk = Dict("candidates" => [Dict("content" => Dict("parts" => [Dict("text" => accumulated)], "role" => "model"))])
            push!(events, "data: $(JSON3.write(chunk))\n\n")
        end
        # Final with usage
        final = Dict("candidates" => [Dict("content" => Dict("parts" => [Dict("text" => text)], "role" => "model"), "finishReason" => "STOP")],
            "usageMetadata" => Dict("promptTokenCount" => tokens_in, "candidatesTokenCount" => tokens_out, "totalTokenCount" => tokens_in + tokens_out))
        push!(events, "data: $(JSON3.write(final))\n\n")
        return join(events)
    else
        return JSON3.write(Dict(
            "candidates" => [Dict("content" => Dict("parts" => [Dict("text" => text)], "role" => "model"), "finishReason" => "STOP")],
            "usageMetadata" => Dict("promptTokenCount" => tokens_in, "candidatesTokenCount" => tokens_out, "totalTokenCount" => tokens_in + tokens_out)
        ))
    end
end

function build_responses_api_response(text, model, stream)
    resp_id, msg_id = "resp_" * gen_id(), "msg_" * gen_id()
    tokens_in, tokens_out = length(split(text)) + 5, length(split(text))
    
    if stream
        events = String[]
        push!(events, "event: response.created\ndata: $(JSON3.write(Dict("type" => "response.created", 
            "response" => Dict("id" => resp_id, "status" => "in_progress", "model" => model))))\n\n")
        for char in text
            push!(events, "event: response.output_text.delta\ndata: $(JSON3.write(Dict("type" => "response.output_text.delta",
                "item_id" => msg_id, "output_index" => 0, "content_index" => 0, "delta" => string(char))))\n\n")
        end
        push!(events, "event: response.completed\ndata: $(JSON3.write(Dict("type" => "response.completed",
            "response" => Dict("id" => resp_id, "status" => "completed", "model" => model,
                "usage" => Dict("input_tokens" => tokens_in, "output_tokens" => tokens_out, "total_tokens" => tokens_in + tokens_out)))))\n\n")
        return join(events)
    else
        return JSON3.write(Dict(
            "id" => resp_id, "object" => "response", "created_at" => round(Int, time()), "status" => "completed", "model" => model,
            "output" => [Dict("type" => "message", "id" => msg_id, "status" => "completed", "role" => "assistant",
                "content" => [Dict("type" => "output_text", "text" => text, "annotations" => [])])],
            "usage" => Dict("input_tokens" => tokens_in, "output_tokens" => tokens_out, "total_tokens" => tokens_in + tokens_out,
                "input_tokens_details" => Dict("cached_tokens" => 0), "output_tokens_details" => Dict("reasoning_tokens" => 0))
        ))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Request handler
# ─────────────────────────────────────────────────────────────────────────────

function echo_handler(req::HTTP.Request)
    req.method != "POST" && return HTTP.Response(405, "Method Not Allowed")
    
    body = JSON3.read(String(req.body))
    input_text = extract_input(body)
    model = get(body, :model, "echo-test")
    stream = get(body, :stream, false)
    echo_text = "Echo: $input_text"
    
    # Route by endpoint
    response_body = if contains(req.target, "/chat/completions")
        build_chat_completion_response(echo_text, model, stream)
    elseif contains(req.target, "/v1/messages")
        build_anthropic_response(echo_text, model, stream)
    elseif contains(req.target, "/models/") && contains(req.target, "Content")
        build_gemini_response(echo_text, model, stream)
    elseif contains(req.target, "/responses")
        build_responses_api_response(echo_text, model, stream)
    else
        return HTTP.Response(404, "Unknown endpoint: $(req.target)")
    end
    
    headers = stream ? ["Content-Type" => "text/event-stream", "Cache-Control" => "no-cache"] : ["Content-Type" => "application/json"]
    return HTTP.Response(200, headers; body = response_body)
end

println("""
╔══════════════════════════════════════════════════════════════════════╗
║  Multi-Schema Echo Server running on http://localhost:$PORT           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Endpoints:                                                          ║
║    POST /v1/chat/completions  → ChatCompletionSchema                 ║
║    POST /v1/messages          → AnthropicSchema                      ║
║    POST /models/{m}:generate* → GeminiSchema                         ║
║    POST /responses            → ResponseSchema                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Test with:                                                          ║
║    aigen("echo_chat:test", "Hello")       # ChatCompletion           ║
║    aigen("echo_anthropic:test", "Hello")  # Anthropic                ║
║    aigen("echo_gemini:test", "Hello")     # Gemini                   ║
║    aigen("echo_responses:test", "Hello")  # Responses API            ║
╚══════════════════════════════════════════════════════════════════════╝
""")
HTTP.serve(echo_handler, "0.0.0.0", PORT)

