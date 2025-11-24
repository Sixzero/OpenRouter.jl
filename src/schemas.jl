"""
Abstract base type for all API schemas.
"""
abstract type AbstractSchema end

"""
Abstract base type for request schemas.
"""
abstract type AbstractRequestSchema <: AbstractSchema end

"""
Abstract base type for response schemas.
"""
abstract type AbstractResponseSchema <: AbstractSchema end

"""
Standard OpenAI-compatible chat completion request schema.
This is the default schema used by most providers.
"""
struct ChatCompletionSchema <: AbstractRequestSchema
    endpoint::String
    
    ChatCompletionSchema() = new("/chat/completions")
end

"""
Build messages array for ChatCompletionSchema.
"""
function build_messages(::ChatCompletionSchema, prompt, sys_msg)
    msgs = normalize_messages(prompt, sys_msg)
    return to_openai_messages(msgs)
end

"""
Build the request payload for ChatCompletionSchema.
"""
function build_payload(::ChatCompletionSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool = false; cache=nothing, kwargs...)
    messages = build_messages(ChatCompletionSchema(), prompt, sys_msg)
    
    payload = Dict{String, Any}(
        "model" => model_id,
        "messages" => messages
    )
    
    # Add stream parameter only if true
    stream && (payload["stream"] = true)
    
    # Add any additional kwargs
    for (k, v) in kwargs
        payload[string(k)] = v
    end
    
    return payload
end

"""
Build the URL for ChatCompletionSchema.
"""
function build_url(schema::ChatCompletionSchema, base_url::AbstractString, model_id::AbstractString, stream::Bool = false)
    return "$(base_url)$(schema.endpoint)"
end

"""
Extract response content for ChatCompletionSchema.
"""
function extract_content(::ChatCompletionSchema, result::Dict)
    if haskey(result, "choices") && length(result["choices"]) > 0
        choice = result["choices"][1]
        if haskey(choice, "message") && haskey(choice["message"], "content")
            return choice["message"]["content"]
        end
    end
    error("Unexpected response format from API: $result")
end

"""
Standard OpenAI-compatible response schema.
"""
struct ChatCompletionResponseSchema <: AbstractResponseSchema end

"""
Extract full response for ChatCompletionResponseSchema.
"""
function extract_response(::ChatCompletionResponseSchema, result::Dict)
    return result
end

"""
Anthropic Claude-style request schema.
"""
struct AnthropicSchema <: AbstractRequestSchema
    endpoint::String
    
    AnthropicSchema() = new("/v1/messages")
end

"""
Build messages array for AnthropicSchema.

Returns a tuple: (messages, system_content)
"""
function build_messages(::AnthropicSchema, prompt, sys_msg)
    normalized = normalize_messages(prompt, sys_msg)

    # Extract first SystemMessage content, if any
    system_content = nothing
    for m in normalized
        if m isa SystemMessage
            system_content = m.content
            break
        end
    end

    # Convert all messages; SystemMessages are ignored by to_anthropic_messages
    msgs = to_anthropic_messages(normalized)
    return msgs, system_content
end

"""
Build the request payload for AnthropicSchema.
"""
function build_payload(::AnthropicSchema, prompt, model_id::AbstractString, sys_msg,
                       stream::Bool = false; max_tokens::Int=1000,
                       cache::Union{Nothing,Symbol}=nothing,
                       kwargs...)
    # normalize + get system
    normalized = normalize_messages(prompt, sys_msg)
    system_content = nothing
    for m in normalized
        if m isa SystemMessage
            system_content = m.content
            break
        end
    end

    # convert messages, applying cache markers if requested
    messages = to_anthropic_messages(normalized; cache)
    
    payload = Dict{String, Any}(
        "model" => model_id,
        "max_tokens" => max_tokens,
        "messages" => messages
    )

    # Add top-level system prompt if present (optionally cached)
    if system_content !== nothing
        if cache !== nothing && (cache == :system || cache == :all || cache == :all_but_last)
            payload["system"] = Any[Dict("type" => "text",
                                         "text" => system_content,
                                         "cache_control" => Dict("type" => "ephemeral"))]
        else
            payload["system"] = system_content
        end
    end
    
    # Add stream parameter only if true
    stream && (payload["stream"] = true)
    
    # Add any additional kwargs
    for (k, v) in kwargs
        v === nothing && continue
        payload[string(k)] = v
    end
    
    return payload
end

"""
Build the URL for AnthropicSchema.
"""
function build_url(schema::AnthropicSchema, base_url::AbstractString, model_id::AbstractString, stream::Bool = false)
    return "$(base_url)$(schema.endpoint)"
end

"""
Extract response content for AnthropicSchema.
"""
function extract_content(::AnthropicSchema, result::Dict)
    if haskey(result, "content") && length(result["content"]) > 0
        content = result["content"][1]
        if haskey(content, "text")
            return content["text"]
        end
    end
    error("Unexpected response format from Anthropic API")
end

"""
Google Gemini-style request schema.
"""
struct GeminiSchema <: AbstractRequestSchema
    endpoint::String
    
    GeminiSchema() = new("/models/{model}:generateContent")
end

"""
Build contents array for GeminiSchema.

Returns a tuple: (contents, system_instruction)
"""
function build_messages(::GeminiSchema, prompt, sys_msg)
    normalized = normalize_messages(prompt, sys_msg)

    # Extract first SystemMessage content, if any
    system_instruction = nothing
    for m in normalized
        if m isa SystemMessage
            system_instruction = Dict("parts" => Any[Dict("text" => m.content)])
            break
        end
    end

    # Convert all messages; SystemMessages are ignored by to_gemini_contents
    contents = to_gemini_contents(normalized)
    return contents, system_instruction
end

"""
Build the request payload for GeminiSchema.
"""
function build_payload(::GeminiSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool = false; cache=nothing, kwargs...)
    contents, system_instruction = build_messages(GeminiSchema(), prompt, sys_msg)
    
    payload = Dict{String, Any}(
        "contents" => contents
    )

    # Add system instruction if present
    if system_instruction !== nothing
        payload["system_instruction"] = system_instruction
    end
    
    generation_config = Dict{String, Any}()
    user_thinking_config = Dict{String, Any}()

    for (k, v) in kwargs
        v === nothing && continue
        sk = string(k)
        if sk == "temperature"
            generation_config["temperature"] = v
        elseif sk == "top_p" || sk == "topP"
            generation_config["topP"] = v
        elseif sk == "top_k" || sk == "topK"
            generation_config["topK"] = v
        elseif sk == "max_output_tokens" || sk == "maxOutputTokens"
            generation_config["maxOutputTokens"] = v
        elseif sk == "presence_penalty" || sk == "presencePenalty"
            generation_config["presencePenalty"] = v
        elseif sk == "frequency_penalty" || sk == "frequencyPenalty"
            generation_config["frequencyPenalty"] = v
        elseif sk == "response_mime_type" || sk == "responseMimeType"
            generation_config["responseMimeType"] = v
        elseif sk == "response_schema" || sk == "responseSchema"
            generation_config["responseSchema"] = v
        elseif sk == "response_json_schema" || sk == "responseJsonSchema"
            generation_config["responseJsonSchema"] = v
        elseif sk == "stop_sequences" || sk == "stopSequences"
            generation_config["stopSequences"] = v
        elseif sk == "thinkingConfig"
            if v isa AbstractDict
                empty!(user_thinking_config)
                for (tk, tv) in v
                    user_thinking_config[string(tk)] = tv
                end
            else
                @warn "thinkingConfig must be an AbstractDict, got $(typeof(v)). Ignoring."
            end
        elseif sk in ["candidateCount", "seed", "responseLogprobs", "logprobs",
                      "enableEnhancedCivicAnswers", "speechConfig",
                      "imageConfig", "mediaResolution"]
            generation_config[sk] = v
        else
            payload[sk] = v
        end
    end
    
    # Always include thoughts; add budget 0 only if no thinkingLevel
    thinking_config = Dict{String, Any}("include_thoughts" => true)
    isempty(user_thinking_config) || merge!(thinking_config, user_thinking_config)
    
    has_thinking_level = haskey(thinking_config, "thinkingLevel")
    if !has_thinking_level && !haskey(thinking_config, "thinkingBudget") && !occursin("pro", model_id)
        thinking_config["thinkingBudget"] = 0
    end
    
    generation_config["thinkingConfig"] = thinking_config
    
    payload["generationConfig"] = generation_config
    
    return payload
end

"""
Build the URL for GeminiSchema (handles model parameter substitution and streaming).
"""
function build_url(schema::GeminiSchema, base_url::AbstractString, model_id::AbstractString, stream::Bool = false)
    endpoint = replace(schema.endpoint, "{model}" => model_id)
    if stream
        endpoint = replace(endpoint, "generateContent" => "streamGenerateContent") * "?alt=sse"
    end
    return "$(base_url)$(endpoint)"
end

"""
Extract response content for GeminiSchema.
"""
function extract_content(::GeminiSchema, result::Dict)
    if haskey(result, "candidates") && length(result["candidates"]) > 0
        candidate = result["candidates"][1]
        if haskey(candidate, "content") && haskey(candidate["content"], "parts")
            parts = candidate["content"]["parts"]
            for part in parts
                # Only extract non-reasoning text
                if !get(part, "thought", false) && haskey(part, "text")
                    return part["text"]
                end
            end
        end
    end
    error("Unexpected response format from Gemini API")
end

"""
Extract reasoning content from API response based on schema.
Returns nothing if schema doesn't support reasoning or no reasoning found.
"""
function extract_reasoning(::ChatCompletionSchema, result::Dict)
    return nothing  # OpenAI doesn't have separate reasoning field
end

function extract_reasoning(::AnthropicSchema, result::Dict)
    if haskey(result, "content") && length(result["content"]) > 0
        for content_block in result["content"]
            if get(content_block, "type", nothing) == "thinking"
                return get(content_block, "thinking", nothing)
            end
        end
    end
    return nothing
end

function extract_reasoning(::GeminiSchema, result::Dict)
    if haskey(result, "candidates") && length(result["candidates"]) > 0
        candidate = result["candidates"][1]
        if haskey(candidate, "content") && haskey(candidate["content"], "parts")
            parts = candidate["content"]["parts"]
            reasoning_parts = String[]
            for part in parts
                if get(part, "thought", false) && haskey(part, "text")
                    push!(reasoning_parts, part["text"])
                end
            end
            return isempty(reasoning_parts) ? nothing : join(reasoning_parts)
        end
    end
    return nothing
end

"""
Extract finish reason from API response based on schema.
"""
function extract_finish_reason(::ChatCompletionSchema, result::Dict)
    if haskey(result, "choices") && length(result["choices"]) > 0
        choice = result["choices"][1]
        return get(choice, "finish_reason", nothing)
    end
    return nothing
end

function extract_finish_reason(::AnthropicSchema, result::Dict)
    return get(result, "stop_reason", nothing)
end

function extract_finish_reason(::GeminiSchema, result::Dict)
    if haskey(result, "candidates") && length(result["candidates"]) > 0
        candidate = result["candidates"][1]
        return get(candidate, "finishReason", nothing)
    end
    return nothing
end
