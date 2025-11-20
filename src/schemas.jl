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
function build_payload(::ChatCompletionSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool = false; kwargs...)
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
    error("Unexpected response format from API")
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
"""
function build_messages(::AnthropicSchema, prompt, sys_msg)
    msgs = normalize_messages(prompt, sys_msg)
    return to_anthropic_messages(msgs)
end

"""
Build the request payload for AnthropicSchema.
"""
function build_payload(::AnthropicSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool = false; max_tokens::Int=1000, kwargs...)
    messages = build_messages(AnthropicSchema(), prompt, sys_msg)
    
    payload = Dict{String, Any}(
        "model" => model_id,
        "max_tokens" => max_tokens,
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
"""
function build_messages(::GeminiSchema, prompt, sys_msg)
    # We treat sys_msg as separate system_instruction; here we only convert prompt/messages.
    msgs = normalize_messages(prompt, nothing)
    return to_gemini_contents(msgs)
end

"""
Build the request payload for GeminiSchema.
"""
function build_payload(::GeminiSchema, prompt, model_id::AbstractString, sys_msg, stream::Bool = false; kwargs...)
    contents = build_messages(GeminiSchema(), prompt, sys_msg)
    
    payload = Dict{String, Any}(
        "contents" => contents
    )

    if sys_msg !== nothing
        payload["system_instruction"] = isa(sys_msg, AbstractString) ?
            Dict("parts" => Any[Dict("text" => sys_msg)]) :
            sys_msg
    end
    
    # Note: Google doesn't use stream=true in payload, it uses different endpoint
    
    # Add generation config if provided
    generation_config = Dict{String, Any}()
    for (k, v) in kwargs
        if string(k) in ["temperature", "topP", "topK", "maxOutputTokens"]
            generation_config[string(k)] = v
        else
            payload[string(k)] = v
        end
    end
    
    if !isempty(generation_config)
        payload["generationConfig"] = generation_config
    end
    
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
            if length(parts) > 0 && haskey(parts[1], "text")
                return parts[1]["text"]
            end
        end
    end
    error("Unexpected response format from Gemini API")
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
