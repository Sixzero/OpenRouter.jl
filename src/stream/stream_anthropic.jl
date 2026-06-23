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
Supports text and tool_use content blocks.
"""
function build_response_body(schema::AnthropicSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing

    response = nothing
    usage = nothing
    content_buf = IOBuffer()
    # Accumulate tool_use blocks: index => Dict with :id, :name, :input_json (string)
    tool_blocks = Dict{Int, Dict{Symbol, Any}}()
    # Accumulate thinking blocks: index => IOBuffer of :thinking deltas
    thinking_blocks = Dict{Int, IOBuffer}()
    current_block_index = -1
    current_block_type = nothing

    for chunk in cb.chunks
        if isnothing(chunk.json)
            verbose && !isempty(strip(chunk.data)) &&
                @warn "Skipping Anthropic chunk with unparsed JSON (dropped delta?)" raw=chunk.data
            continue
        end

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

        # Track content block boundaries
        if chunk_type == "content_block_start"
            current_block_index = get(chunk.json, :index, current_block_index + 1)
            content_block = get(chunk.json, :content_block, Dict())
            current_block_type = get(content_block, :type, nothing)

            if current_block_type == "tool_use"
                tool_blocks[current_block_index] = Dict{Symbol, Any}(
                    :id => get(content_block, :id, ""),
                    :name => get(content_block, :name, ""),
                    :input_json => IOBuffer()
                )
            elseif current_block_type == "thinking"
                buf = IOBuffer()
                initial = get(content_block, :thinking, "")
                !isempty(initial) && write(buf, initial)
                thinking_blocks[current_block_index] = buf
            end
        end

        if chunk_type == "content_block_delta"
            delta = get(chunk.json, :delta, Dict())
            delta_type = get(delta, :type, nothing)

            if delta_type == "text_delta"
                text = get(delta, :text, nothing)
                !isnothing(text) && write(content_buf, text)
            elseif delta_type == "input_json_delta"
                partial = get(delta, :partial_json, nothing)
                if !isnothing(partial) && haskey(tool_blocks, current_block_index)
                    write(tool_blocks[current_block_index][:input_json], partial)
                end
            elseif delta_type == "thinking_delta"
                t = get(delta, :thinking, nothing)
                if !isnothing(t) && haskey(thinking_blocks, current_block_index)
                    write(thinking_blocks[current_block_index], t)
                end
            end
        end
    end

    if !isnothing(response)
        response isa JSON3.Object && (response = copy(response))

        # Build content array with thinking, text, and tool_use blocks (preserve order: thinking first)
        content = Any[]

        for idx in sort(collect(keys(thinking_blocks)))
            push!(content, Dict(:type => "thinking", :thinking => String(take!(thinking_blocks[idx]))))
        end

        text_content = String(take!(content_buf))
        !isempty(text_content) && push!(content, Dict(:type => "text", :text => text_content))

        for idx in sort(collect(keys(tool_blocks)))
            tb = tool_blocks[idx]
            json_str = String(take!(tb[:input_json]))
            input = if isempty(json_str)
                Dict()
            else
                try
                    JSON3.read(json_str, Dict)
                catch e
                    @warn "Incomplete tool_use JSON for block $idx (name=$(tb[:name])); a streamed delta was likely dropped or the stream closed early" exception=e bytes=sizeof(json_str) raw=json_str
                    Dict()
                end
            end
            push!(content, Dict(:type => "tool_use", :id => tb[:id], :name => tb[:name], :input => input))
        end

        response[:content] = content
        !isnothing(usage) && (response[:usage] = usage)
    end

    return response
end