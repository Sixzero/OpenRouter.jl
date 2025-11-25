"""
    ResponseSchema <: AbstractRequestSchema

Schema for OpenAI's Responses API (v1/responses endpoint).
Used by newer models like gpt-5.1 and o-series models.
"""
struct ResponseSchema <: AbstractRequestSchema end

# Build URL for Responses API
function build_url(schema::ResponseSchema, base_url::String, model_id::AbstractString, stream::Bool)
    return "$base_url/responses"
end

# Build payload for Responses API
function build_payload(schema::ResponseSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool; reasoning=nothing, kwargs...)
    payload = Dict{String, Any}(
        "model" => model_id,
        "stream" => stream,
    )
    
    # Normalize and partition messages
    normalized = normalize_messages(prompt, sys_msg)
    system_msgs = filter(m -> m isa SystemMessage, normalized)
    input_msgs = filter(m -> !(m isa SystemMessage), normalized)
    
    # Build input items
    payload["input"] = [message_to_response_input(msg) for msg in input_msgs]
    
    # Add combined system instructions if any
    if !isempty(system_msgs)
        payload["instructions"] = join([m.content for m in system_msgs], "\n\n")
    end
    
    # Default reasoning config, merge user's reasoning if provided
    default_reasoning = Dict("summary" => "detailed")
    payload["reasoning"] = isnothing(reasoning) ? default_reasoning : merge(default_reasoning, reasoning)
    
    # Forward supported parameters
    for (k, v) in kwargs
        if k in (:temperature, :top_p, :max_output_tokens, :tools, :tool_choice, 
                 :parallel_tool_calls, :metadata, :text, :truncation,
                 :service_tier, :safety_identifier, :prompt_cache_key, :prompt_cache_retention)
            payload[string(k)] = v
        end
    end
    
    return payload
end

# Convert message types to Response API input format
function message_to_response_input(msg::UserMessage)
    return Dict(
        "type" => "message",
        "role" => "user",
        "content" => msg.content
    )
end

function message_to_response_input(msg::AIMessage)
    return Dict(
        "type" => "message",
        "role" => "assistant",
        "content" => msg.content
    )
end

function message_to_response_input(msg::AbstractString)
    return Dict(
        "type" => "message",
        "role" => "user",
        "content" => msg
    )
end

# Extract reasoning from Response API
function extract_reasoning(schema::ResponseSchema, response::Dict)
    output = get(response, "output", [])
    reasoning_parts = String[]
    
    for item in output
        if get(item, "type", nothing) == "reasoning"
            content = get(item, "content", [])
            for reasoning_item in content
                if get(reasoning_item, "type", nothing) == "reasoning_text"
                    text = get(reasoning_item, "text", nothing)
                    !isnothing(text) && push!(reasoning_parts, text)
                end
            end
        end
    end
    
    return isempty(reasoning_parts) ? nothing : join(reasoning_parts, "\n")
end

# Extract finish reason
function extract_finish_reason(schema::ResponseSchema, response::Dict)
    return get(response, "status", nothing)
end

# Extract tokens from Response API usage format
function extract_tokens(schema::ResponseSchema, response::Union{Dict, JSON3.Object})
    usage = get(response, "usage", nothing)
    isnothing(usage) && return nothing
    
    prompt_tokens = get(usage, "input_tokens", 0)
    completion_tokens = get(usage, "output_tokens", 0)
    total_tokens = get(usage, "total_tokens", prompt_tokens + completion_tokens)
    
    # Extract detailed token counts
    input_details = get(usage, "input_tokens_details", Dict())
    output_details = get(usage, "output_tokens_details", Dict())
    
    return TokenCounts(
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        total_tokens = total_tokens,
        input_cache_read = get(input_details, "cached_tokens", 0),
        internal_reasoning = get(output_details, "reasoning_tokens", 0)
    )
end

# Extract content from non-streaming response
function extract_content(schema::ResponseSchema, response::Dict)
    output = get(response, "output", [])
    content_parts = String[]
    
    for item in output
        if get(item, "type", nothing) == "message"
            content_array = get(item, "content", [])
            for content_item in content_array
                if get(content_item, "type", nothing) == "output_text"
                    text = get(content_item, "text", nothing)
                    !isnothing(text) && push!(content_parts, text)
                end
            end
        end
    end
    
    return isempty(content_parts) ? nothing : join(content_parts, "")
end

