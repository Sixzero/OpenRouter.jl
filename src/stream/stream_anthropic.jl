# Custom methods for Anthropic streaming

"""
    is_start(schema::AnthropicSchema, chunk::AbstractStreamChunk; kwargs...)

Check if streaming has started for Anthropic format.
"""
@inline function is_start(schema::AnthropicSchema, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        chunk_type = get(chunk.json, :type, nothing)
        return chunk_type == "message_start" || chunk_type == :message_start
    end
    false
end

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

function acc_tokens(schema::AnthropicSchema, accumulator::TokenCounts, new_tokens::TokenCounts)
    # Anthropic sends cumulative counts, so replace rather than add
    return new_tokens
end

# Anthropic sends usage early with low completion_tokens (<=3)
is_usr_meta(schema::AnthropicSchema, tokens::TokenCounts) = 
    tokens.prompt_tokens > 0 && tokens.completion_tokens <= 3

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
        
        chunk_type = get(chunk.json, :type, nothing)
        
        # Core message body from message_start
        if chunk_type == "message_start" && haskey(chunk.json, :message)
            response = chunk.json[:message] |> copy
            usage = get(response, :usage, Dict())
        end
        
        # Update stop reason and usage from message_delta
        if chunk_type == "message_delta"
            delta = get(chunk.json, :delta, Dict())
            response = isnothing(response) ? copy(delta) : merge(response, delta)
            
            # Extract usage from message_delta (this is where final token counts come)
            chunk_usage = get(chunk.json, :usage, nothing)
            if !isnothing(chunk_usage)
                usage = isnothing(usage) ? copy(chunk_usage) : merge(usage, chunk_usage)
            end
        end

        # Load text chunks from content blocks
        if chunk_type == "content_block_start" || 
           chunk_type == "content_block_delta" || 
           chunk_type == "content_block_stop"
            
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