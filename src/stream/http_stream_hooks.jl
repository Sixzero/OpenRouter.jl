"""
    RunInfo(; creation_time=time(), inference_start=nothing, last_message_time=nothing, stop_sequence=nothing)

Tracks run statistics and metadata during the streaming process.

# Fields
- `creation_time`: When the callback was created
- `inference_start`: When the model started processing
- `last_message_time`: Timestamp of the last received message
- `stop_sequence`: The sequence that caused the generation to stop (if any). For OpenAI this can be:
  - A specific stop sequence provided in the chunk's delta.stop_sequence
  - "stop" if finish_reason is "stop"
  For Anthropic this is the stop_sequence provided in the chunk.

# Timing Methods
- `get_total_elapsed(info)`: Get total elapsed time since callback creation
- `get_inference_elapsed(info)`: Get elapsed time for inference phase only
"""
@kwdef mutable struct RunInfo
    creation_time::Float64 = time()
    inference_start::Union{Float64,Nothing} = nothing
    last_message_time::Union{Float64,Nothing} = nothing
    stop_sequence::Union{String,Nothing} = nothing
end

@kwdef mutable struct RunInfoMeta
    tokens::TokenCounts = TokenCounts()
    cost::Float64 = 0.0
    elapsed::Float64 = 0.0
    model_info::Union{String,Nothing} = nothing
end

# Add utility functions for RunInfo
"""
    get_total_elapsed(info::RunInfo)

Get total elapsed time since callback creation.
Returns time in seconds or nothing if no messages received.
"""
get_total_elapsed(info::RunInfo) = !isnothing(info.last_message_time) ? info.last_message_time - info.creation_time : nothing

"""
    get_inference_elapsed(info::RunInfo)

Get elapsed time for inference (time between first inference and last message).
Returns time in seconds or nothing if inference hasn't started.
"""
get_inference_elapsed(info::RunInfo) = !isnothing(info.inference_start) && !isnothing(info.last_message_time) ?
    info.last_message_time - info.inference_start : nothing

"""
    needs_tool_execution(info::RunInfo)

Check if the run was terminated because the model is requested tool execution (with stop_sequence).
"""
function needs_tool_execution(info::RunInfo)
    return !isnothing(info.stop_sequence)
end

# Color constants
const REASONING_COLOR = "\e[94m"  # Light blue ANSI escape code
const RESET_COLOR = "\e[0m"      # Reset ANSI escape code

"""
    HttpStreamHooks

A stream callback that combines token counting with customizable hooks for various events.
"""
@kwdef mutable struct HttpStreamHooks <: AbstractLLMStream
    out::IO = stdout
    schema::Union{AbstractRequestSchema, Nothing} = nothing
    chunks::Vector{<:StreamChunk} = StreamChunk[]
    verbose::Bool = false
    throw_on_error::Bool = false
    kwargs::NamedTuple = NamedTuple()
    run_info::RunInfo = RunInfo()
    model::Union{String,Nothing} = nothing
    total_tokens::TokenCounts = TokenCounts()  # Now using the universal TokenCounts
    
    # Internal state
    in_reasoning_mode::Bool = false
    
    # Provider endpoint for cost calculation
    provider_endpoint::Union{ProviderEndpoint, Nothing} = nothing
    
    # Hooks with default implementations
    content_formatter::Function = identity
    reasoning_formatter::Function = text -> "$(REASONING_COLOR)$text$(RESET_COLOR)"
    on_meta_usr::Function = (tokens, cost=0.0, elapsed=nothing) -> format_user_meta(tokens, cost, elapsed)
    on_meta_ai::Function = (tokens, cost=0.0, elapsed=nothing) -> format_ai_meta(tokens, cost, elapsed)
    on_error::Function = e -> format_error_message(e)
    on_done::Function = () -> nothing
    on_start::Function = () -> nothing
    on_stop_sequence::Function = identity
end

# Default formatters
function format_user_meta(tokens::TokenCounts, cost::Float64, elapsed::Union{Float64, Nothing})
    return "User tokens: $(tokens.prompt_tokens), Cost: \$$(round(cost, digits=6))"
end

function format_ai_meta(tokens::TokenCounts, cost::Float64, elapsed::Union{Float64, Nothing})
    elapsed_str = elapsed !== nothing ? ", Time: $(round(elapsed, digits=2))s" : ""
    return "AI tokens: $(tokens.completion_tokens), Cost: \$$(round(cost, digits=6))$elapsed_str"
end

function format_error_message(e::Exception)
    return "Stream error: $(string(e))"
end

function configure_stream_callback!(cb::HttpStreamHooks, schema::AbstractRequestSchema, provider_info::ProviderInfo, provider_endpoint::ProviderEndpoint)
    cb.schema = schema
    cb.provider_endpoint = provider_endpoint
    return cb
end

# Extract model from chunk based on schema
function extract_model_from_chunk(schema::AbstractRequestSchema, chunk::StreamChunk)
    if !isnothing(chunk.json)
        return get(chunk.json, :model, nothing)
    end
    return nothing
end

# Extract reasoning content for thinking models
function extract_reasoning_from_chunk(schema::AnthropicSchema, chunk::StreamChunk)
    isnothing(chunk.json) && return nothing
    
    chunk_type = get(chunk.json, :type, nothing)
    
    if chunk_type == "content_block_start" || chunk_type == "content_block_stop"
        content_block = get(chunk.json, :content_block, Dict())
        block_type = get(content_block, :type, nothing)
        
        if block_type == "thinking"
            return get(content_block, :thinking, nothing)
        end
    elseif chunk_type == "content_block_delta"
        delta = get(chunk.json, :delta, Dict())
        delta_type = get(delta, :type, nothing)
        
        if delta_type == "thinking_delta"
            return get(delta, :thinking, nothing)
        end
    end
    
    return nothing
end

# Default: no reasoning extraction for other schemas
extract_reasoning_from_chunk(schema::AbstractRequestSchema, chunk::StreamChunk) = nothing

# Extract stop sequence
function extract_stop_sequence_from_chunk(schema::AbstractRequestSchema, chunk::StreamChunk)
    isnothing(chunk.json) && return nothing
    
    # Schema-specific stop sequence extraction
    if schema isa ChatCompletionSchema
        choices = get(chunk.json, :choices, [])
        if !isempty(choices)
            return get(choices[1], :finish_reason, nothing)
        end
    elseif schema isa AnthropicSchema
        return get(chunk.json, :stop_reason, nothing)
    elseif schema isa GeminiSchema
        candidates = get(chunk.json, :candidates, [])
        if !isempty(candidates)
            return get(candidates[1], :finishReason, nothing)
        end
    end
    
    return nothing
end

# Handle token metadata with schema-specific dispatch
function handle_token_metadata_for_schema(schema::AbstractRequestSchema, cb::HttpStreamHooks, 
                                         tokens::TokenCounts, cost::Float64, elapsed::Float64)
    # Determine if this is user or AI metadata based on token types
    if tokens.prompt_tokens > 0 && tokens.completion_tokens == 0
        msg = cb.on_meta_usr(tokens, cost, elapsed)
    else
        msg = cb.on_meta_ai(tokens, cost, elapsed)
    end
    
    isa(msg, AbstractString) && println(cb.out, msg)
end

# Extract tokens from chunk using existing schema methods
function extract_tokens_from_chunk(schema::AbstractRequestSchema, chunk::StreamChunk)
    isnothing(chunk.json) && return nothing
    
    # Use existing extract_tokens method from costs_tokens.jl
    return extract_tokens(schema, chunk.json)
end

# Main callback implementation - compatible with existing pattern
function callback(cb::HttpStreamHooks, chunk::StreamChunk; kwargs...)
    # Early return if no json
    isnothing(chunk.json) && return nothing

    # Handle message start
    if get(chunk.json, :type, nothing) == "message_start"
        cb.run_info.inference_start = time()
        msg = cb.on_start()
        isa(msg, AbstractString) && println(cb.out, msg)
    end

    # Extract model info if needed
    if isnothing(cb.model) && !isnothing(cb.schema)
        cb.model = extract_model_from_chunk(cb.schema, chunk)
    end

    # Handle content
    try
        if !isnothing(cb.schema)
            if (reasoning = extract_reasoning_from_chunk(cb.schema, chunk)) !== nothing
                formatted = cb.reasoning_formatter(reasoning)
                isa(formatted, AbstractString) && !cb.in_reasoning_mode && print(cb.out, "$(REASONING_COLOR)")
                cb.in_reasoning_mode = true
                isa(formatted, AbstractString) && print(cb.out, formatted)
            elseif (text = extract_content(cb.schema, chunk; kwargs...)) !== nothing
                formatted = cb.content_formatter(text)
                if cb.in_reasoning_mode
                    isa(formatted, AbstractString) && print(cb.out, "$(RESET_COLOR)\n\n")
                    cb.in_reasoning_mode = false
                end
                isa(formatted, AbstractString) && print(cb.out, formatted)
            end
        end
    catch e
        msg = cb.on_error(e)
        isa(msg, AbstractString) && println(stderr, msg)
        cb.throw_on_error && rethrow(e)
    end

    # Store stop sequence if present
    if !isnothing(cb.schema) && (stop_seq = extract_stop_sequence_from_chunk(cb.schema, chunk)) !== nothing
        cb.run_info.stop_sequence = stop_seq
        cb.on_stop_sequence(stop_seq)
        msg = cb.on_done()
        isa(msg, AbstractString) && println(cb.out, msg)
    end

    # Handle token metadata with schema-specific dispatch
    if !isnothing(cb.schema) && (tokens = extract_tokens_from_chunk(cb.schema, chunk)) !== nothing
        cb.total_tokens = cb.total_tokens + tokens
        
        # Use existing calculate_cost function with provider endpoint
        cost = 0.0
        if !isnothing(cb.provider_endpoint)
            calculated_cost = calculate_cost(cb.provider_endpoint, cb.total_tokens)
            cost = calculated_cost !== nothing ? calculated_cost : 0.0
        end
        
        cb.run_info.last_message_time = time()
        elapsed = get_total_elapsed(cb.run_info)

        handle_token_metadata_for_schema(cb.schema, cb, tokens, cost, elapsed !== nothing ? elapsed : 0.0)
    end

    # Handle completion
    if get(chunk.json, :type, nothing) in ("message_end", "message_stop")
        msg = cb.on_done()
        isa(msg, AbstractString) && println(cb.out, msg)
    end
    
    return nothing
end

# Add the missing streamed_request! method for HttpStreamHooks
function streamed_request!(cb::HttpStreamHooks, url, headers, input::String; kwargs...)
    verbose = get(kwargs, :verbose, false) || cb.verbose
    resp = HTTP.open("POST", url, headers; kwargs...) do stream
        write(stream, input)
        HTTP.closewrite(stream)
        response = HTTP.startread(stream)

        # Validate content type
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
                push!(cb.chunks, chunk)
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