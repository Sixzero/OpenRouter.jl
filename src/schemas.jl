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
Build the request payload for ChatCompletionSchema.
"""
function build_payload(::ChatCompletionSchema, prompt::String, model_id::String; kwargs...)
    payload = Dict{String, Any}(
        "model" => model_id,
        "messages" => [Dict("role" => "user", "content" => prompt)]
    )
    
    # Add any additional kwargs
    for (k, v) in kwargs
        payload[string(k)] = v
    end
    
    return payload
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
Build the request payload for AnthropicSchema.
"""
function build_payload(::AnthropicSchema, prompt::String, model_id::String; max_tokens::Int=1000, kwargs...)
    payload = Dict{String, Any}(
        "model" => model_id,
        "max_tokens" => max_tokens,
        "messages" => [Dict("role" => "user", "content" => prompt)]
    )
    
    # Add any additional kwargs
    for (k, v) in kwargs
        payload[string(k)] = v
    end
    
    return payload
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
    
    GeminiSchema() = new("/v1/models/{model}:generateContent")
end

"""
Build the request payload for GeminiSchema.
"""
function build_payload(::GeminiSchema, prompt::String, model_id::String; kwargs...)
    payload = Dict{String, Any}(
        "contents" => [Dict(
            "parts" => [Dict("text" => prompt)]
        )]
    )
    
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
Get the appropriate schema for a provider.
"""
function get_provider_schema(provider_name::String)::AbstractRequestSchema
    provider_lower = lowercase(provider_name)
    
    if provider_lower in ["anthropic", "claude"]
        return AnthropicSchema()
    elseif provider_lower in ["google", "gemini"]
        return GeminiSchema()
    else
        # Default to ChatCompletion for most providers
        return ChatCompletionSchema()
    end
end