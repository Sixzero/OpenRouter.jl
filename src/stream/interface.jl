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

"""
    configure_stream_callback!(cb::AbstractLLMStream, schema::AbstractRequestSchema, provider_info::ProviderInfo, provider_endpoint::ProviderEndpoint)

Configure stream callback with schema and provider information.
For HttpStreamHooks, also sets up pricing for accurate cost calculation.
"""
function configure_stream_callback!(cb::AbstractLLMStream, schema::AbstractRequestSchema, provider_info::ProviderInfo, provider_endpoint::ProviderEndpoint)
    cb.schema = schema
    return cb
end


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

Base.empty!(cb::AbstractLLMStream) = empty!(cb.chunks)
Base.push!(cb::AbstractLLMStream, chunk::StreamChunk) = push!(cb.chunks, chunk)
Base.isempty(cb::AbstractLLMStream) = isempty(cb.chunks)
Base.length(cb::AbstractLLMStream) = length(cb.chunks)