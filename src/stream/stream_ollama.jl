# Custom methods for Ollama native streaming.
#
# Ollama does NOT use SSE: it streams newline-delimited JSON (one JSON object per
# line) with Content-Type `application/json`. So we override `extract_chunks` to
# split on newlines instead of SSE `data:` framing, and mark the schema as
# non-SSE so `_open_sse_stream` skips the `text/event-stream` assertion.

is_sse_stream(::OllamaSchema) = false

# Split NDJSON: each complete line is one chunk; an unterminated trailing line spills over.
function extract_chunks(::OllamaSchema, blob::AbstractString;
        spillover::AbstractString="", verbose::Bool=false, kwargs...)
    full = spillover * blob
    chunks = StreamChunk[]
    next_spillover = ""
    lines = split(full, '\n')
    # Last element is spillover unless the blob ended exactly on a newline.
    if !isempty(lines) && !endswith(full, '\n')
        next_spillover = String(pop!(lines))
    end
    for line in lines
        s = strip(line)
        isempty(s) && continue
        json = try
            JSON3.read(s)
        catch
            verbose && @warn "Cannot parse Ollama NDJSON line: $(repr(s))"
            continue   # a complete-but-malformed line is dropped, not surfaced as a null chunk
        end
        push!(chunks, StreamChunk(nothing, String(s), json))
    end
    return chunks, next_spillover
end

@inline function is_done(::OllamaSchema, chunk::AbstractStreamChunk; kwargs...)
    isnothing(chunk.json) && return false
    return get(chunk.json, :done, false) == true
end

@inline function is_start(::OllamaSchema, chunk::AbstractStreamChunk; kwargs...)
    isnothing(chunk.json) && return false
    return haskey(chunk.json, :message)
end

@inline function extract_content(::OllamaSchema, chunk::AbstractStreamChunk; kwargs...)
    isnothing(chunk.json) && return nothing
    msg = get(chunk.json, :message, nothing)
    isnothing(msg) && return nothing
    content = get(msg, :content, nothing)
    return (isnothing(content) || isempty(content)) ? nothing : content
end

@inline function extract_reasoning_from_chunk(::OllamaSchema, chunk::AbstractStreamChunk)
    isnothing(chunk.json) && return nothing
    msg = get(chunk.json, :message, nothing)
    isnothing(msg) && return nothing
    thinking = get(msg, :thinking, nothing)
    return (isnothing(thinking) || isempty(thinking)) ? nothing : thinking
end

extract_stop_sequence_from_chunk(::OllamaSchema, chunk::AbstractStreamChunk) =
    isnothing(chunk.json) ? nothing : get(chunk.json, :done_reason, nothing)

# Reassemble a non-streaming `/api/chat` response from the NDJSON chunks.
function build_response_body(::OllamaSchema, cb::AbstractLLMStream; verbose::Bool=false, kwargs...)
    isempty(cb.chunks) && return nothing
    content = IOBuffer()
    thinking = IOBuffer()
    tool_calls = Any[]
    role = "assistant"
    response = Dict{String,Any}()
    for chunk in cb.chunks
        isnothing(chunk.json) && continue
        # Keep latest top-level metadata (done, done_reason, *_count, ...) with string keys.
        for (k, v) in pairs(chunk.json)
            k === :message && continue
            response[string(k)] = v
        end
        msg = get(chunk.json, :message, nothing)
        isnothing(msg) && continue
        r = get(msg, :role, nothing); isnothing(r) || (role = r)
        c = get(msg, :content, nothing); isnothing(c) || print(content, c)
        t = get(msg, :thinking, nothing); isnothing(t) || print(thinking, t)
        tc = get(msg, :tool_calls, nothing); isnothing(tc) || append!(tool_calls, tc)
    end
    message = Dict{String,Any}("role" => role, "content" => String(take!(content)))
    th = String(take!(thinking)); isempty(th) || (message["thinking"] = th)
    isempty(tool_calls) || (message["tool_calls"] = tool_calls)
    response["message"] = message
    return response
end
