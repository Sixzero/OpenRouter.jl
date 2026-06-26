# Default methods for streaming interface

# Idle timeout (seconds) for streaming reads. This library provides the
# MECHANISM only; the timeout POLICY belongs to the caller, who alone knows
# whether a long silence is a dead stream or a reasoning model thinking.
# Disabled by default (0) so we never silently abort a healthy generation;
# callers opt in via the `stream_idle_timeout` kwarg.
const DEFAULT_STREAM_IDLE_TIMEOUT = 0.0

"""
    StreamIdleTimeoutError(timeout)

Thrown when no bytes arrive on a streaming response for `timeout` seconds.
This is an *idle* timeout (it resets on every received chunk), so it never
interrupts an actively-streaming generation — only a genuinely stalled one.
"""
struct StreamIdleTimeoutError <: Exception
    timeout::Float64
end
Base.showerror(io::IO, e::StreamIdleTimeoutError) =
    print(io, "StreamIdleTimeoutError: no data received for $(e.timeout)s (stream stalled)")

# Abort an in-flight read by closing the underlying connection/socket.
# For an `HTTP.Stream`, `close(stream)` only marks the wrapper and does NOT
# interrupt a blocked `readavailable`; closing `stream.stream` (the
# Connection/TCPSocket) does. `hasproperty` guards against HTTP.jl renaming the
# internal field — fall back to closing the wrapper itself.
_abort_read!(stream::HTTP.Stream) =
    close(hasproperty(stream, :stream) ? stream.stream : stream)
_abort_read!(stream) = close(stream)

"""
    readavailable_with_idle_timeout(stream, timeout; fired) -> Vector{UInt8}

Like `readavailable(stream)` but if no byte arrives within `timeout` seconds the
underlying connection is closed via `_abort_read!` (which is what unblocks the
in-flight read) and `fired[]` is set. `timeout <= 0` disables the guard. The
deadline applies to this single read, so a caller looping over chunks gets a
per-chunk idle timeout that resets on every chunk.

This does not itself throw — the abort makes `readavailable` return/throw, and
the caller decides what to do based on `fired[]` (see `_open_sse_stream`). That
keeps timeout detection in one place across the `HTTP.open` boundary, where HTTP
cleanup would otherwise mask a thrown error with an `EOFError`.
"""
function readavailable_with_idle_timeout(stream, timeout::Real; fired::Ref{Bool}=Ref(false))
    timeout <= 0 && return readavailable(stream)
    timer = Timer(timeout) do _
        fired[] = true
        try
            _abort_read!(stream)
        catch e
            @debug "failed to abort stalled stream read" exception=(e, catch_backtrace())
        end
    end
    try
        return readavailable(stream)
    finally
        close(timer)
    end
end


@inline function _has_double_newline_end(s::AbstractString)
    endswith(s, "\n\n") || endswith(s, "\r\n\r\n")
end

@inline function _split_sse_messages(full_blob::AbstractString)
    # Support both Unix and HTTP CRLF double-newline boundaries.
    msgs = String[]
    start = firstindex(full_blob)
    i = start
    last = lastindex(full_blob)

    while i <= last
        # Look for \n\n
        if i < last && full_blob[i] == '\n' && full_blob[nextind(full_blob, i)] == '\n'
            push!(msgs, full_blob[start:(i-1)])
            i = nextind(full_blob, nextind(full_blob, i))
            start = i
            continue
        end
        # Look for \r\n\r\n
        if i < last && full_blob[i] == '\r'
            j = nextind(full_blob, i)
            if j < last && full_blob[j] == '\n'
                k = nextind(full_blob, j)
                if k < last && full_blob[k] == '\r'
                    l = nextind(full_blob, k)
                    if l <= last && full_blob[l] == '\n'
                        push!(msgs, full_blob[start:(i-1)])
                        i = nextind(full_blob, l)
                        start = i
                        continue
                    end
                end
            end
        end
        i = nextind(full_blob, i)
    end

    if start <= last
        push!(msgs, full_blob[start:end])
    elseif isempty(msgs)
        push!(msgs, "")
    end

    return msgs
end

"""
    extract_chunks(schema::AbstractRequestSchema, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)

Extract chunks from received SSE blob. Correctly implements SSE spec field parsing.
"""
@inline function extract_chunks(schema::AbstractRequestSchema, blob::AbstractString;
        spillover::AbstractString = "", verbose::Bool = false, kwargs...)

    # Handle spillover from previous incomplete message
    full_blob = spillover * blob

    # Split messages on either \n\n or \r\n\r\n
    messages = _split_sse_messages(full_blob)

    # Check if last message is incomplete (no trailing double newline)
    next_spillover = ""
    if !_has_double_newline_end(full_blob) && !isempty(messages)
        next_spillover = pop!(messages)
        verbose && !isempty(strip(next_spillover)) &&
            @info "Incomplete message detected, spillover: $(repr(next_spillover))"
    end

    chunks = StreamChunk[]

    for message in messages
        isempty(strip(message)) && continue

        # Parse SSE fields
        event_name = nothing
        data_parts = String[]

        for line in split(message, '\n')
            try
                line = rstrip(line, '\r')  # Handle \r\n

                # Handle comments (lines starting with ":")
                startswith(line, ":") && continue

                # Parse field:value lines
                colon_pos = findfirst(':', line)
                isnothing(colon_pos) && continue

                field_name = line[1:(colon_pos - 1)]
                field_value = line[(colon_pos + 1):end]

                # Strip UTF-8 BOM from first field name if present
                if !isempty(field_name) && field_name[1] == '\ufeff'
                    field_name = field_name[nextind(field_name, 1):end]
                end

                # Remove leading space from field value if present
                startswith(field_value, " ") && (field_value = field_value[2:end])

                # Handle data fields
                if field_name == "data"
                    push!(data_parts, field_value)
                elseif field_name == "event" && !isempty(field_value)
                    event_name = Symbol(field_value)
                end
            catch e
                verbose && @warn "Malformed SSE line ignored: $(repr(line)). Error: $e"
                continue
            end
        end

        isempty(data_parts) && continue

        # Join multiple data lines with newlines (SSE spec)
        raw_data = join(data_parts, '\n')

        # JSON detection and parsing
        parsed_json = if !isempty(strip(raw_data))
            stripped = strip(raw_data)
            is_json = (startswith(stripped, '{') && endswith(stripped, '}')) ||
                      (startswith(stripped, '[') && endswith(stripped, ']'))
            if is_json
                try
                    JSON3.read(raw_data)
                catch e
                    verbose && @warn "Cannot parse JSON: $(repr(raw_data))"
                    nothing
                end
            else
                nothing
            end
        else
            nothing
        end

        push!(chunks, StreamChunk(event_name, raw_data, parsed_json))
    end

    return chunks, next_spillover
end

function is_done(schema::AbstractRequestSchema, chunk::AbstractStreamChunk; kwargs...)
    throw(ArgumentError("is_done is not implemented for schema $(typeof(schema))"))
end

# Whether the schema's stream is Server-Sent Events (`data:`/`text/event-stream`).
# Ollama overrides this to false (it streams newline-delimited JSON).
is_sse_stream(::AbstractRequestSchema) = true

function extract_content(schema::AbstractRequestSchema, chunk::AbstractStreamChunk; kwargs...)
    throw(ArgumentError("extract_content is not implemented for schema $(typeof(schema))"))
end

function print_content(out::Any, text::AbstractString; kwargs...)
    throw(ArgumentError("print_content is not implemented for sink $(typeof(out))"))
end

"""
    print_content(out::IO, text::AbstractString; kwargs...)

Print content to IO output stream.
"""
@inline function print_content(out::IO, text::AbstractString; kwargs...)
    print(out, text)
end

"""
    print_content(out::Channel, text::AbstractString; kwargs...)

Print content to Channel.
"""
@inline function print_content(out::Channel, text::AbstractString; kwargs...)
    put!(out, text)
end

"""
    print_content(out::Nothing, text::Any; kwargs...)

Do nothing if output stream is `nothing`.
"""
@inline function print_content(out::Nothing, text::AbstractString; kwargs...)
    return nothing
end

"""
    callback(cb::AbstractLLMStream, chunk::AbstractStreamChunk; kwargs...)

Process chunk and print it. Wrapper for:
- extract content from chunk using `extract_content`
- print content to output stream using `print_content`
"""
@inline function callback(cb::AbstractLLMStream, chunk::AbstractStreamChunk; kwargs...)
    @warn "Unimplemented callback function: $(typeof(cb))"
    return nothing
end

"""
    handle_error_message(chunk::AbstractStreamChunk; kwargs...)

Handle error messages from streaming response. Always throws on error.
"""
@inline function handle_error_message(chunk::AbstractStreamChunk; kwargs...)
    if chunk.event == :error ||
       (isnothing(chunk.event) && !isnothing(chunk.json) && haskey(chunk.json, :error))
        
        has_error_dict = !isnothing(chunk.json) && get(chunk.json, :error, nothing) isa AbstractDict
        
        error_str = if has_error_dict
            join(["$(titlecase(string(k))): $(v)" for (k, v) in pairs(chunk.json.error)], ", ")
        else
            string(chunk.data)
        end
        
        error("Error detected in streaming response: $(error_str)")
    end
    return nothing
end

"Extract a human-readable error message from an API error body (JSON `error`/`message`, else raw)."
function stream_error_message(body::AbstractString)
    isempty(strip(body)) && return "<empty response body>"
    try
        json = JSON3.read(body)
        if haskey(json, :error)
            detail = json.error
            return detail isa AbstractDict ? string(get(detail, :message, detail)) : string(detail)
        end
        haskey(json, :message) && return string(json.message)
    catch
    end
    return body
end

"""
    throw_stream_http_error(response, stream, input)

Read an error response body and throw a descriptive `HTTP.RequestError`. Called for any
`status >= 400` *before* the event-stream content-type check, so providers that return an
error with a missing or non-stream Content-Type (e.g. z.ai 429) surface the real message
instead of a misleading content-type assertion failure.
"""
function throw_stream_http_error(response, stream, input::AbstractString)
    body = String(read(stream))
    HTTP.closeread(stream)
    response.status == 400 && @error "API 400: request body snippet" body_snippet=input[1:min(500,end)]
    throw(HTTP.RequestError(response, "API Error ($(response.status)): $(stream_error_message(body))"))
end

"""
    _open_sse_stream(cb, url, headers, input; verbose, kwargs...) -> HTTP.Response

Open a streaming POST request, validate it's an event-stream, then read chunks
with a per-chunk idle timeout (`stream_idle_timeout`, default disabled) and feed
each parsed chunk to the schema hooks + `callback(cb, chunk)`. Shared by all
`streamed_request!` callback types — they differ only in their post-processing.
"""
function _open_sse_stream(cb::AbstractLLMStream, url, headers, input::String; verbose::Bool, kwargs...)
    idle_timeout = get(kwargs, :stream_idle_timeout, get(cb.kwargs, :stream_idle_timeout, DEFAULT_STREAM_IDLE_TIMEOUT))
    http_kwargs = Base.structdiff(NamedTuple(kwargs), (; stream_idle_timeout=nothing))
    # Closing the socket on idle makes HTTP cleanup throw an EOFError that masks
    # the StreamIdleTimeoutError; `idle_fired` lets us re-surface it.
    idle_fired = Ref(false)
    try HTTP.open("POST", url, headers; http_kwargs...) do stream
        write(stream, input)
        HTTP.closewrite(stream)
        response = HTTP.startread(stream)

        # On error status, surface the real API error first — error responses may
        # omit Content-Type (e.g. z.ai 429), so don't gate this on the header.
        response.status >= 400 && throw_stream_http_error(response, stream, input)

        # Success path: SSE schemas must declare a single event-stream Content-Type.
        # Non-SSE schemas (e.g. Ollama NDJSON, `application/json`) skip this check.
        if is_sse_stream(cb.schema)
            content_type = [header[2] for header in response.headers if lowercase(header[1]) == "content-type"]
            @assert length(content_type) == 1 "Content-Type header must be present and unique"
            @assert occursin("text/event-stream", lowercase(content_type[1])) """
                Content-Type header should include text/event-stream.
                Received: $(content_type[1])
                Status: $(response.status)
                Headers: $(response.headers)
                Body: $(String(response.body))
                Please check model and that stream=true is set.
                """
        end

        isdone = false
        spillover = ""
        while !eof(stream) && !isdone
            masterchunk = String(readavailable_with_idle_timeout(stream, idle_timeout; fired=idle_fired))
            # The idle abort can make the read return empty instead of throwing;
            # treat that as a stall, not a clean end-of-stream.
            idle_fired[] && throw(StreamIdleTimeoutError(Float64(idle_timeout)))
            chunks, spillover = extract_chunks(cb.schema, masterchunk; verbose, spillover, cb.kwargs...)
            for chunk in chunks
                verbose && @debug "Chunk Data: $(chunk.data)"
                handle_error_message(chunk; verbose, cb.kwargs...)          # always throws on error
                is_done(cb.schema, chunk; verbose, cb.kwargs...) && (isdone = true)
                callback(cb, chunk; verbose, cb.kwargs...)
                push!(cb, chunk)
            end
        end
        HTTP.closeread(stream)
    end
    catch e
        idle_fired[] && !(e isa StreamIdleTimeoutError) && throw(StreamIdleTimeoutError(Float64(idle_timeout)))
        rethrow()
    end
end

"""
    streamed_request!(cb::AbstractLLMStream, url, headers, input; kwargs...)

End-to-end wrapper for POST streaming requests.
Modifies callback object (`cb.chunks`) in-place and returns response object.
"""
function streamed_request!(cb::AbstractLLMStream, url, headers, input::IOBuffer; kwargs...)
    streamed_request!(cb, url, headers, String(take!(input)); kwargs...)
end
function streamed_request!(cb::AbstractLLMStream, url, headers, input::IO; kwargs...)
    streamed_request!(cb, url, headers, read(input); kwargs...)
end
function streamed_request!(cb::AbstractLLMStream, url, headers, input::Dict; kwargs...)
    streamed_request!(cb, url, headers, String(JSON3.write(input)); kwargs...)
end