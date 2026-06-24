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
    resp = _open_sse_stream(cb, url, headers, input; verbose, kwargs...)

    # Aesthetic newline for stdout
    cb.out == stdout && (println(); flush(stdout))

    # Build response body
    body = build_response_body(cb.schema, cb; verbose, cb.kwargs...)
    resp.body = JSON3.write(body)

    return resp
end