# Default methods for streaming interface


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
@inline function print_content(out::Nothing, text::Any; kwargs...)
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
        
        throw("Error detected in streaming response: $(error_str)")
    end
    return nothing
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