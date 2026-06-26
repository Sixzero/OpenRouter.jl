"""
    OllamaSchema <: AbstractRequestSchema

Schema for Ollama's native API (`/api/chat`). Unlike `ChatCompletionSchema`
(Ollama's OpenAI-compatible `/v1/chat/completions` shim), this targets the
native endpoint, which exposes `thinking` reasoning, native `tool_calls`,
`format` structured outputs, `keep_alive`, and NDJSON streaming.

Docs: https://docs.ollama.com/api/chat
"""
struct OllamaSchema <: AbstractRequestSchema end

# Ollama tool format: { type: "function", function: { name, description, parameters } } (same as OpenAI)
convert_tool(::OllamaSchema, tool::Tool) = convert_tool(_ccs, tool)

build_url(::OllamaSchema, base_url::AbstractString, model_id::AbstractString, stream::Bool=false) = "$(base_url)/chat"

# Ollama messages are OpenAI-shaped, but images go in a top-level `images` array of
# bare base64 strings (no data: prefix) instead of OpenAI's content-parts.
function build_messages(::OllamaSchema, prompt, sys_msg)
    msgs = to_openai_messages(normalize_messages(prompt, sys_msg))
    for m in msgs
        content = get(m, "content", nothing)
        content isa AbstractVector || continue
        texts = String[]
        images = String[]
        for part in content
            part isa AbstractDict || continue
            if get(part, "type", "") == "text"
                push!(texts, get(part, "text", ""))
            elseif get(part, "type", "") == "image_url"
                url = get(get(part, "image_url", Dict()), "url", "")
                push!(images, replace(url, r"^data:[^,]*," => ""))
            end
        end
        m["content"] = join(texts, "\n")
        isempty(images) || (m["images"] = images)
    end
    return msgs
end

function build_payload(::OllamaSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool=false; kwargs...)
    payload = Dict{String,Any}(
        "model" => model_id,
        "messages" => build_messages(OllamaSchema(), prompt, sys_msg),
        "stream" => stream,
    )
    # Ollama groups sampling params (temperature, top_p, num_ctx, ...) under `options`.
    options = Dict{String,Any}()
    for (k, v) in kwargs
        v === nothing && continue
        if k == :tools
            payload["tools"] = convert_tools(OllamaSchema(), v)
        elseif k in (:format, :think, :keep_alive, :options)
            payload[string(k)] = v
        else
            options[string(k)] = v
        end
    end
    isempty(options) || (payload["options"] = merge(options, get(payload, "options", Dict{String,Any}())))
    return payload
end

function extract_content(::OllamaSchema, result::Dict)
    msg = get(result, "message", nothing)
    msg === nothing && error("Unexpected response format from Ollama API: $result")
    return get(msg, "content", "")
end

function extract_reasoning(::OllamaSchema, result::Dict)
    msg = get(result, "message", nothing)
    msg === nothing && return nothing
    thinking = get(msg, "thinking", nothing)
    return (thinking === nothing || isempty(thinking)) ? nothing : thinking
end

extract_finish_reason(::OllamaSchema, result::Dict) = get(result, "done_reason", nothing)

function extract_tool_calls(::OllamaSchema, result::Dict)
    msg = get(result, "message", nothing)
    msg === nothing && return nothing
    calls = get(msg, "tool_calls", nothing)
    calls === nothing && return nothing
    out = Dict{String,Any}[]
    for (i, tc) in enumerate(calls)
        fn = tc["function"]
        args = get(fn, "arguments", Dict())
        push!(out, Dict{String,Any}(
            "id" => get(tc, "id", "ollama_$i"),
            "type" => "function",
            "function" => Dict("name" => fn["name"],
                               "arguments" => args isa AbstractString ? args : JSON3.write(args))
        ))
    end
    return isempty(out) ? nothing : out
end

# Ollama reports raw token counts; no cache breakdown.
function extract_tokens(::OllamaSchema, result::Union{Dict,JSON3.Object})
    haskey(result, "prompt_eval_count") || haskey(result, "eval_count") || return nothing
    prompt_tokens = get(result, "prompt_eval_count", 0)
    completion_tokens = get(result, "eval_count", 0)
    return TokenCounts(
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        total_tokens = prompt_tokens + completion_tokens,
    )
end
