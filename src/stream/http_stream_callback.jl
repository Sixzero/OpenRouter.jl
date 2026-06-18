using HTTP, JSON3

"""
    HttpStreamCallback

HTTP-based streaming callback that prints content to output stream.
When streaming completes, builds response body from chunks as if it was a normal API response.
"""
@kwdef mutable struct HttpStreamCallback{T1 <: Any} <: AbstractLLMStream
    out::T1 = stdout
    schema::Union{AbstractRequestSchema, Nothing} = nothing
    chunks::Vector{<:StreamChunk} = StreamChunk[]
    verbose::Bool = false
    kwargs::NamedTuple = NamedTuple()
end

function Base.show(io::IO, cb::HttpStreamCallback)
    print(io, "HttpStreamCallback(out=$(cb.out), schema=$(cb.schema), chunks=$(length(cb.chunks)) items, $(cb.verbose ? "verbose" : "silent"))")
end

"""
    callback(cb::AbstractLLMStream, chunk::AbstractStreamChunk; kwargs...)

Process chunk and print it. Wrapper for:
- extract content from chunk using `extract_content`
- print content to output stream using `print_content`
"""
@inline function callback(cb::HttpStreamCallback, chunk::AbstractStreamChunk; kwargs...)
    processed_text = extract_content(cb.schema, chunk; kwargs...)
    isnothing(processed_text) && return nothing
    print_content(cb.out, processed_text; kwargs...)
    return nothing
end
function streamed_request!(cb::HttpStreamCallback, url, headers, input::String; kwargs...)
    verbose = get(kwargs, :verbose, false) || cb.verbose
    resp = HTTP.open("POST", url, headers; kwargs...) do stream
        write(stream, input)
        HTTP.closewrite(stream)
        response = HTTP.startread(stream)

        # On error status, surface the real API error first — error responses may
        # omit Content-Type (e.g. z.ai 429), so don't gate this on the header.
        response.status >= 400 && throw_stream_http_error(response, stream, input)

        # Validate content type (success path: must be a single event-stream header)
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

        isdone = false
        spillover = ""
        
        while !eof(stream) && !isdone
            masterchunk = String(readavailable(stream))
            chunks, spillover = extract_chunks(cb.schema, masterchunk; verbose, spillover, cb.kwargs...)

            for chunk in chunks
                verbose && @debug "Chunk Data: $(chunk.data)"
                
                # Handle errors (always throw)
                handle_error_message(chunk; verbose, cb.kwargs...)
                
                # Check for termination
                is_done(cb.schema, chunk; verbose, cb.kwargs...) && (isdone = true)
                # Trigger callback
                callback(cb, chunk; verbose, cb.kwargs...)
                
                # Store chunk
                push!(cb, chunk)
            end
        end
        HTTP.closeread(stream)
    end
    
    # Aesthetic newline for stdout
    cb.out == stdout && (println(); flush(stdout))

    # Build response body
    body = build_response_body(cb.schema, cb; verbose, cb.kwargs...)
    resp.body = JSON3.write(body)

    return resp
end