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
    extract_content(schema::GeminiSchema, chunk::AbstractStreamChunk; kwargs...)

Extract content from Gemini chunk.
"""
@inline function extract_content(schema::GeminiSchema, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json) && haskey(chunk.json, :candidates)
        candidates = chunk.json[:candidates]
        if !isempty(candidates)
            candidate = candidates[1]
            if haskey(candidate, :content) && haskey(candidate[:content], :parts)
                parts = candidate[:content][:parts]
                if !isempty(parts) && haskey(parts[1], :text)
                    return parts[1][:text]
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
    final_candidate = nothing
    
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
                        push!(content_parts, part[:text])
                    end
                end
            end
        end
    end
    
    if !isnothing(response) && !isnothing(final_candidate) && !isempty(content_parts)
        full_content = join(content_parts)
        
        # Build final candidate with accumulated content and final state
        final_candidate_dict = Dict(final_candidate)
        final_candidate_dict[:content] = Dict(
            :parts => [Dict(:text => full_content)],
            :role => get(get(final_candidate_dict, :content, Dict()), :role, "model")
        )
        
        response[:candidates] = [final_candidate_dict]
    end
    
    return response
end