# This file defines the core streaming interface for OpenRouter.jl
#
# The goal is to enable custom callbacks for streaming LLM APIs using HTTP,
# and reconstruct the response body in a standard way to mimic non-streaming responses

using HTTP, JSON3

# ## Interface
# It all revolves around the `HttpStreamCallback` object, 
# which holds individual "chunks" (StreamChunk) and processing logic
#
# Top-level interface that wraps the HTTP.POST request and handles streaming
function streamed_request! end
# Extract chunks from received SSE blob. Returns list of `StreamChunk`
function extract_chunks end
# Check if the stream is done for a given schema
function is_done end
# Main function for user logic - provided each "chunk" to process
function callback end
# Extract content from chunk for a given schema
function extract_content end
# Print content to different "sinks"
function print_content end
# Build response body from chunks to mimic standard API response
function build_response_body end

## Default implementations
"""
    StreamChunk

A chunk of streaming data. A message is composed of multiple chunks.

# Fields
- `event`: The event name
- `data`: The data chunk
- `json`: The JSON object or `nothing` if chunk doesn't contain JSON
"""
@kwdef struct StreamChunk{T1 <: AbstractString, T2 <: Union{JSON3.Object, Nothing}} <: AbstractStreamChunk
    event::Union{Symbol, Nothing} = nothing
    data::T1 = ""
    json::T2 = nothing
end

function Base.show(io::IO, chunk::StreamChunk)
    data_preview = length(chunk.data) > 10 ? "$(first(chunk.data, 10))..." : chunk.data
    json_keys = !isnothing(chunk.json) ? join(keys(chunk.json), ", ", " and ") : "-"
    print(io, "StreamChunk(event=$(chunk.event), data=$(data_preview), json keys=$(json_keys))")
end

"""
    HttpStreamCallback

HTTP-based streaming callback that prints content to output stream.
When streaming completes, builds response body from chunks as if it was a normal API response.

# Fields
- `out`: Output stream (e.g., `stdout` or pipe)
- `schema`: Request schema determining API format
- `chunks`: List of received `StreamChunk` chunks  
- `verbose`: Whether to print verbose information
- `kwargs`: Custom keyword arguments

# Example
```julia
# Simple usage - stream to stdout
callback = HttpStreamCallback(; out=stdout, schema=ChatCompletionSchema())
response = aigen("Count to 10", "OpenAI:gpt-4o-mini"; stream_callback=callback)

# Record all chunks for inspection
callback = HttpStreamCallback(; schema=ChatCompletionSchema())
response = aigen("Count to 10", "OpenAI:gpt-4o-mini"; stream_callback=callback)
# Inspect with callback.chunks
```
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

Base.empty!(cb::AbstractLLMStream) = empty!(cb.chunks)
Base.push!(cb::AbstractLLMStream, chunk::StreamChunk) = push!(cb.chunks, chunk)
Base.isempty(cb::AbstractLLMStream) = isempty(cb.chunks)
Base.length(cb::AbstractLLMStream) = length(cb.chunks)