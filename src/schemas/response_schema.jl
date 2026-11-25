"""
    ResponseSchema <: AbstractRequestSchema

Schema for OpenAI's Responses API (v1/responses endpoint).
Used by newer models like gpt-5.1 and o-series models.
"""
struct ResponseSchema <: AbstractRequestSchema end

# ResponseSchema format: { type: "function", name, description, parameters } (flat, no function wrapper)
function convert_tool(::ResponseSchema, tool::Tool)
    Dict{String,Any}(
        "type" => "function",
        "name" => tool.name,
        "description" => tool.description,
        "parameters" => tool.parameters
    )
end

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
    
    # Build input items (all methods return Vector for uniform flattening).
    # ToolMessage attachments flush once after a run of consecutive ToolMessages so a user
    # message never splits parallel function_call_outputs.
    input_items = Any[]
    pending_tool_parts = Any[]
    for (i, msg) in enumerate(input_msgs)
        if msg isa ToolMessage
            push!(input_items, Dict{String,Any}(
                "type" => "function_call_output", "call_id" => msg.tool_call_id, "output" => msg.content))
            append!(pending_tool_parts, _response_attachments(msg.image_data, msg.document_data))
            next = i < length(input_msgs) ? input_msgs[i + 1] : nothing
            if !(next isa ToolMessage) && !isempty(pending_tool_parts)
                push!(input_items, Dict{String,Any}("type" => "message", "role" => "user", "content" => pending_tool_parts))
                pending_tool_parts = Any[]
            end
        else
            append!(input_items, msg isa AIMessage ? message_to_response_input(msg; model_id) : message_to_response_input(msg))
        end
    end
    payload["input"] = input_items
    
    # Add combined system instructions if any
    if !isempty(system_msgs)
        payload["instructions"] = join([m.content for m in system_msgs], "\n\n")
    end
    
    # Default reasoning config, merge user's reasoning if provided
    default_reasoning = Dict("summary" => "detailed")
    payload["reasoning"] = isnothing(reasoning) ? default_reasoning : merge(default_reasoning, reasoning)
    # Encrypted reasoning comes back on each `reasoning` item and is replayed on the next
    # turn (stateless: works with store=false/ZDR, compaction, forks, provider switches).
    payload["include"] = Any["reasoning.encrypted_content"]
    
    # Forward supported parameters (convert tools if present)
    for (k, v) in kwargs
        v === nothing && continue
        if k in (:temperature, :top_p, :max_output_tokens,
                 :parallel_tool_calls, :metadata, :text, :truncation,
                 :service_tier, :safety_identifier, :prompt_cache_key, :prompt_cache_retention,
                 :store, :include)
            payload[string(k)] = v
        elseif k in (:max_tokens, :max_completion_tokens)
            haskey(payload, "max_output_tokens") || (payload["max_output_tokens"] = v)
        elseif k == :reasoning_effort
            haskey(payload["reasoning"], "effort") || (payload["reasoning"] = merge(payload["reasoning"], Dict("effort" => v)))
        elseif k == :tool_choice
            # ChatCompletion wraps named tool_choice in {"type":"function","function":{"name":…}};
            # Responses API expects flat {"type":"function","name":…}.
            if v isa AbstractDict && haskey(v, "function")
                payload["tool_choice"] = Dict{String,Any}("type" => "function", "name" => v["function"]["name"])
            else
                payload["tool_choice"] = v
            end
        elseif k == :tools
            payload["tools"] = convert_tools(ResponseSchema(), v)
        end
    end
    
    # Route every turn of one conversation to the same prompt-cache shard (and let
    # CLIProxyAPI scope Codex caching on /v1/responses, where headers alone don't).
    if !haskey(payload, "prompt_cache_key")
        sid = current_session_id()
        isempty(sid) && (sid = get(ENV, "TODOFORAI_TODO_ID", ""))
        isempty(sid) || (payload["prompt_cache_key"] = first(sid, 64))
    end

    return payload
end

function _response_attachments(image_data, document_data)
    parts = Any[]
    for img in something(image_data, ())
        push!(parts, isnothing(extract_image_attributes(img)) ?
            Dict{String,Any}("type" => "input_text", "text" => INVALID_IMAGE_NOTICE) :
            Dict{String,Any}("type" => "input_image", "image_url" => img))
    end
    for doc in something(document_data, ())
        push!(parts, isnothing(extract_document_attributes(doc)) ?
            Dict{String,Any}("type" => "input_text", "text" => INVALID_DOCUMENT_NOTICE) :
            Dict{String,Any}("type" => "input_file", "filename" => "document.pdf", "file_data" => doc))
    end
    parts
end

# Convert message types to Response API input format (all return Vector for uniform flattening)
function message_to_response_input(msg::UserMessage)
    att = _response_attachments(msg.image_data, msg.document_data)
    if !isempty(att)
        content = Any[]
        !isempty(msg.content) && push!(content, Dict{String,Any}("type" => "input_text", "text" => msg.content))
        append!(content, att)
        return [Dict("type" => "message", "role" => "user", "content" => content)]
    end
    [Dict("type" => "message", "role" => "user", "content" => msg.content)]
end

function message_to_response_input(msg::AIMessage; model_id=nothing)
    items = Any[]
    # Reasoning items precede the output they led to, as in the original response. No `id`:
    # with store=false a referenced id is "not found"; the encrypted content is self-contained.
    if (r = replay_reasoning_items(msg, model_id)) !== nothing
        for it in r
            get(it, "type", "") == "reasoning" && push!(items, it)
        end
    end
    !isempty(msg.content) && push!(items, Dict("type" => "message", "role" => "assistant", "content" => msg.content))
    if msg.tool_calls !== nothing
        for tc in msg.tool_calls
            fn = tc["function"]
            push!(items, Dict{String,Any}(
                "type" => "function_call", "call_id" => tc["id"],
                "name" => fn["name"], "arguments" => isempty(fn["arguments"]) ? "{}" : fn["arguments"]
            ))
        end
    end
    all(i -> get(i, "type", "") == "reasoning", items) &&
        push!(items, Dict("type" => "message", "role" => "assistant", "content" => msg.content))
    return items
end

message_to_response_input(msg::AbstractString) = [Dict(
    "type" => "message", "role" => "user", "content" => msg
)]

# `reasoning` output items that carry encrypted_content, trimmed to what replay needs.
function extract_reasoning_items(::ResponseSchema, response::Dict)
    items = Any[]
    for item in get(response, "output", [])
        get(item, "type", nothing) == "reasoning" || continue
        enc = get(item, "encrypted_content", nothing)
        enc isa AbstractString && !isempty(enc) || continue
        push!(items, Dict{String,Any}("type" => "reasoning", "summary" => something(get(item, "summary", nothing), Any[]), "encrypted_content" => enc))
    end
    isempty(items) ? nothing : items
end

# Extract reasoning from Response API (raw reasoning text, else the reasoning summaries)
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
            if isempty(content)
                for s in something(get(item, "summary", nothing), [])
                    text = get(s, "text", nothing)
                    text isa AbstractString && push!(reasoning_parts, text)
                end
            end
        end
    end
    
    return isempty(reasoning_parts) ? nothing : join(reasoning_parts, "\n")
end

# Extract finish reason
function extract_finish_reason(schema::ResponseSchema, response::Dict)
    status = get(response, "status", nothing)
    if status == "incomplete"
        details = get(response, "incomplete_details", nothing)
        reason = details isa AbstractDict ? get(details, "reason", nothing) : nothing
        reason !== nothing && return reason
    end
    return status
end

# Extract tokens from Response API usage format
function extract_tokens(schema::ResponseSchema, response::Union{Dict, JSON3.Object})
    usage = get(response, "usage", nothing)
    
    # For response.completed events, check nested response.usage
    if usage === nothing
        nested_response = get(response, "response", nothing)
        if nested_response !== nothing
            usage = get(nested_response, "usage", nothing)
        end
    end
    
    isnothing(usage) && return nothing
    
    api_input_tokens = get(usage, "input_tokens", 0)
    completion_tokens = get(usage, "output_tokens", 0)
    total_tokens = get(usage, "total_tokens", api_input_tokens + completion_tokens)
    
    # Extract detailed token counts
    input_details = get(usage, "input_tokens_details", Dict())
    output_details = get(usage, "output_tokens_details", Dict())
    
    # Responses API input_tokens is the TOTAL input (cached included). Normalized
    # TokenCounts contract: prompt_tokens = cache misses only, so subtract cache hits
    # (otherwise cost and context-size consumers double-count cached input).
    input_cache_read = get(input_details, "cached_tokens", 0)
    
    return TokenCounts(
        prompt_tokens = api_input_tokens - input_cache_read,
        completion_tokens = completion_tokens,
        total_tokens = total_tokens,
        input_cache_read = input_cache_read,
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

function extract_tool_calls(::ResponseSchema, result::Dict)
    output = get(result, "output", [])
    tool_calls = Dict{String,Any}[]
    for item in output
        if get(item, "type", nothing) == "function_call"
            push!(tool_calls, Dict{String,Any}(
                "id" => get(item, "call_id", get(item, "id", nothing)),
                "type" => "function",
                "function" => Dict("name" => item["name"], "arguments" => item["arguments"])
            ))
        end
    end
    return isempty(tool_calls) ? nothing : tool_calls
end

"""
Extract generated images from Response API output.
Returns Vector{String} of base64 data URLs or nothing if no images.
"""
function extract_images(::ResponseSchema, response::Dict)
    output = get(response, "output", [])
    images = String[]

    for item in output
        item_type = get(item, "type", nothing)

        # Handle image_generation_call type (direct image output)
        if item_type == "image_generation_call"
            result = get(item, "result", nothing)
            if result !== nothing
                # Result is base64 encoded image data
                push!(images, "data:image/png;base64,$result")
            end
        # Handle message type with output_image content
        elseif item_type == "message"
            content_array = get(item, "content", [])
            for content_item in content_array
                if get(content_item, "type", nothing) == "output_image"
                    image_url = get(content_item, "image_url", nothing)
                    if image_url !== nothing
                        url = get(image_url, "url", nothing)
                        !isnothing(url) && push!(images, url)
                    end
                end
            end
        end
    end

    return isempty(images) ? nothing : images
end

# Default for other schemas - no image extraction
extract_images(::AbstractRequestSchema, response::Dict) = nothing
