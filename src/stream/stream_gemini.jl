# Custom methods for Gemini streaming

@inline function is_done(schema::GeminiSchema, chunk::AbstractStreamChunk; kwargs...)
    verbose = get(kwargs, :verbose, false)
    
    # Gemini typically ends with empty candidates or specific finish reason
    if !isnothing(chunk.json) && haskey(chunk.json, :candidates)
        candidates = chunk.json[:candidates]
        if isempty(candidates)
            verbose && @info "Gemini: Empty candidates detected, marking as done"
            return true
        end
        # Check for finish reason
        first_candidate = get(candidates, 1, Dict())
        finish_reason = get(first_candidate, :finishReason, nothing)
        
        if !isnothing(finish_reason)
            verbose && @info "Gemini: Finish reason detected: $finish_reason"
            return true
        else
            verbose && @info "Gemini: No finish reason in candidate: $(keys(first_candidate))"
        end
    else
        verbose && @info "Gemini: No candidates in chunk JSON: $(keys(chunk.json))"
    end
    return false
end

"""
    extract_reasoning_from_chunk(schema::GeminiSchema, chunk::StreamChunk)

Extract reasoning/thinking content from Gemini chunk (parts with "thought": true).
"""
function extract_reasoning_from_chunk(schema::GeminiSchema, chunk::StreamChunk)
    if !isnothing(chunk.json) && haskey(chunk.json, :candidates)
        candidates = chunk.json[:candidates]
        if !isempty(candidates)
            candidate = candidates[1]
            if haskey(candidate, :content) && haskey(candidate[:content], :parts)
                parts = candidate[:content][:parts]
                for part in parts
                    # Check if this part is marked as thought/reasoning
                    if get(part, :thought, false) && haskey(part, :text)
                        return part[:text]
                    end
                end
            end
        end
    end
    return nothing
end

"""
    extract_content(schema::GeminiSchema, chunk::StreamChunk; kwargs...)

Extract regular (non-reasoning) content from Gemini chunk.
"""
@inline function extract_content(schema::GeminiSchema, chunk::StreamChunk; kwargs...)
    if !isnothing(chunk.json) && haskey(chunk.json, :candidates)
        candidates = chunk.json[:candidates]
        if !isempty(candidates)
            candidate = candidates[1]
            if haskey(candidate, :content) && haskey(candidate[:content], :parts)
                parts = candidate[:content][:parts]
                for part in parts
                    # Only extract text that is NOT marked as thought
                    if !get(part, :thought, false) && haskey(part, :text)
                        return part[:text]
                    end
                end
            end
        end
    end
    return nothing
end

"""
    build_response_body(schema::GeminiSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)

Build response body from chunks to mimic standard Gemini API response.
"""
function build_response_body(schema::GeminiSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    
    response = nothing
    content_parts = String[]
    reasoning_parts = String[]
    final_candidate = nothing
    final_usage_metadata = nothing
    
    for chunk in cb.chunks
        isnothing(chunk.json) && continue
        
        if isnothing(response)
            response = chunk.json |> copy
        end
        
        # Collect content from candidates and track the final candidate state
        if haskey(chunk.json, :candidates) && !isempty(chunk.json[:candidates])
            candidate = chunk.json[:candidates][1]
            final_candidate = candidate  # Keep updating to get the final state
            
            if haskey(candidate, :content) && haskey(candidate[:content], :parts)
                parts = candidate[:content][:parts]
                for part in parts
                    if haskey(part, :text)
                        # Separate reasoning from regular content
                        if get(part, :thought, false)
                            push!(reasoning_parts, part[:text])
                        else
                            push!(content_parts, part[:text])
                        end
                    end
                end
            end
        end
        
        # Track the final usage metadata (appears in last chunk with complete token counts)
        if haskey(chunk.json, :usageMetadata)
            final_usage_metadata = chunk.json[:usageMetadata]
        end
    end
    
    if !isnothing(response) && !isnothing(final_candidate)
        full_content = join(content_parts)
        full_reasoning = join(reasoning_parts)
        
        # Build final candidate with accumulated content and final state
        final_candidate_dict = Dict(final_candidate)
        
        # Build parts array with both reasoning and content if present
        parts = []
        if !isempty(full_reasoning)
            push!(parts, Dict(:text => full_reasoning, :thought => true))
        end
        if !isempty(full_content)
            push!(parts, Dict(:text => full_content))
        end
        
        if !isempty(parts)
            final_candidate_dict[:content] = Dict(
                :parts => parts,
                :role => get(get(final_candidate_dict, :content, Dict()), :role, "model")
            )
        end
        
        response[:candidates] = [final_candidate_dict]
        
        # Preserve the final usage metadata with complete token counts
        if !isnothing(final_usage_metadata)
            response[:usageMetadata] = final_usage_metadata
        end
    end
    
    return response
end