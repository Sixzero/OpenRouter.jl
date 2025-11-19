# Custom methods for Anthropic streaming

@inline function is_done(schema::AnthropicSchema, chunk::AbstractStreamChunk; kwargs...)
    chunk.event == :error || chunk.event == :message_stop
end

"""
    extract_content(schema::AnthropicSchema, chunk::AbstractStreamChunk;
        include_thinking::Bool = true, kwargs...)

Extract content from Anthropic chunk.
"""
function extract_content(schema::AnthropicSchema, chunk::AbstractStreamChunk;
        include_thinking::Bool = true, kwargs...)
    isnothing(chunk.json) && return nothing

    chunk_type = get(chunk.json, :type, nothing)

    # Handle content blocks (start and stop)
    if chunk_type == "content_block_start" || chunk_type == "content_block_stop"
        content_block = get(chunk.json, :content_block, Dict())
        block_type = get(content_block, :type, nothing)

        if block_type == "text"
            return get(content_block, :text, nothing)
        elseif include_thinking && block_type == "thinking"
            return get(content_block, :thinking, nothing)
        end
    elseif chunk_type == "content_block_delta"
        delta = get(chunk.json, :delta, Dict())
        delta_type = get(delta, :type, nothing)

        if delta_type == "text_delta"
            return get(delta, :text, nothing)
        elseif include_thinking && delta_type == "thinking_delta"
            return get(delta, :thinking, nothing)
        end
    end

    return nothing
end

"""
    build_response_body(schema::AnthropicSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)

Build response body from chunks to mimic standard Anthropic API response.

Note: Limited functionality. Does NOT support tool use.
"""
function build_response_body(schema::AnthropicSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    
    response = nothing
    usage = nothing
    content_buf = IOBuffer()
    
    for chunk in cb.chunks
        isnothing(chunk.json) && continue
        
        # Core message body
        if isnothing(response) && chunk.event == :message_start && haskey(chunk.json, :message)
            response = chunk.json[:message] |> copy
            usage = get(response, :usage, Dict())
        end
        
        # Update stop reason and usage
        if chunk.event == :message_delta
            response = isnothing(response) ? get(copy(chunk.json), :delta, Dict()) :
                       merge(response, get(chunk.json, :delta, Dict()))
            usage = isnothing(usage) ? get(copy(chunk.json), :usage, Dict()) :
                    merge(usage, get(chunk.json, :usage, Dict()))
        end

        # Load text chunks
        if chunk.event == :content_block_start ||
           chunk.event == :content_block_delta || chunk.event == :content_block_stop
            
            delta_block = get(chunk.json, :content_block, nothing)
            isnothing(delta_block) && (delta_block = get(chunk.json, :delta, Dict()))
            
            text = get(delta_block, :text, nothing)
            !isnothing(text) && write(content_buf, text)
        end
    end
    
    if !isnothing(response)
        response isa JSON3.Object && (response = copy(response))
        response[:content] = [Dict(:type => "text", :text => String(take!(content_buf)))]
        !isnothing(usage) && (response[:usage] = usage)
    end
    
    return response
end