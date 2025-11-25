# Custom methods for Response API streaming (OpenAI gpt-5, o-series)

"""
    is_done(schema::ResponseSchema, chunk::AbstractStreamChunk; kwargs...)

Check if streaming is done for Response API format.
"""
@inline function is_done(schema::ResponseSchema, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        # Handle done chunks - response.completed signals the end
        chunk_type = get(chunk.json, "type", nothing)
        return chunk_type == "response.completed"
    end
    false
end

"""
    extract_content(schema::ResponseSchema, chunk::AbstractStreamChunk; kwargs...)

Extract content from Response API chunk. 
Only extracts 'delta' to ensure stream consumers don't print duplicate content 
(since 'done' events contain the full text).
"""
@inline function extract_content(schema::ResponseSchema, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        # Only extract deltas for real-time streaming
        if get(chunk.json, "type", nothing) == "response.output_text.delta"
            return get(chunk.json, "delta", nothing)
        end
    end
    return nothing
end

# Extract reasoning content for Response API streaming (thinking models)
function extract_reasoning_from_chunk(schema::ResponseSchema, chunk::StreamChunk)
    isnothing(chunk.json) && return nothing
    
    chunk_type = get(chunk.json, :type, nothing)
    
    # Handle reasoning summary deltas
    if chunk_type == "response.reasoning_summary_text.delta"
        return get(chunk.json, :delta, nothing)
    end
    
    return nothing
end

"""
    build_response_body(schema::ResponseSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)

Build response body from chunks. 
Optimized to find the final `response.completed` object immediately (O(1) effectively),
with a fallback reconstruction for interrupted streams.
"""
function build_response_body(schema::ResponseSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    
    # Fast path: Check for the final response.completed chunk (usually the last one)
    # Use Iterators.reverse to avoid allocating a new array
    for chunk in Iterators.reverse(cb.chunks)
        if !isnothing(chunk.json) && get(chunk.json, "type", nothing) == "response.completed"
            return get(chunk.json, "response", nothing)
        end
    end
    @warn "Probably we didn't receive usage informations."
    # Fallback: Reconstruct from delta chunks (Best Effort for interrupted streams)
    response = Dict{String, Any}(
        "object" => "response",
        "output" => Any[],
        "status" => "incomplete"
    )
    
    # Collect content from delta chunks
    # Map output_index -> content string
    content_map = Dict{Int, String}()
    metadata = Dict{String, Any}()
    
    for chunk in cb.chunks
        isnothing(chunk.json) && continue
        
        chunk_type = get(chunk.json, "type", nothing)
        
        if chunk_type == "response.output_text.delta"
            idx = get(chunk.json, "output_index", 0)
            delta = get(chunk.json, "delta", "")
            content_map[idx] = get(content_map, idx, "") * delta
        end
        
        # Grab metadata from the first available chunk
        if isempty(metadata) && haskey(chunk.json, "item_id")
            for key in ["id", "model", "created_at"]
                if haskey(chunk.json, key)
                    metadata[key] = chunk.json[key]
                end
            end
        end
    end
    
    # Heuristic reconstruction: 
    # Usually index 0 is reasoning (if present) or message, index 1 is message.
    # Without the final schema, we treat them as generic output items.
    final_output = Any[]
    
    for idx in sort(collect(keys(content_map)))
        text = content_map[idx]
        # Simple heuristic: if we have multiple outputs, 0 might be reasoning
        type = (idx == 0 && length(content_map) > 1) ? "reasoning" : "message"
        
        if type == "reasoning"
            push!(final_output, Dict(
                "type" => "reasoning",
                "summary" => [text] # Rough approximation
            ))
        else
            push!(final_output, Dict(
                "type" => "message",
                "role" => "assistant",
                "content" => Any[Dict(
                    "type" => "output_text",
                    "text" => text
                )]
            ))
        end
    end
    
    response["output"] = final_output
    merge!(response, metadata)
    
    return response
end