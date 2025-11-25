"""
Configuration for model calls, including provider/model slug and call parameters.

# Supported Parameters by Schema

## ChatCompletionSchema (OpenAI-compatible)
Common parameters supported by most OpenAI-compatible providers:
- `temperature::Float64`: Sampling temperature (0.0-2.0, default varies by model)
- `max_tokens::Int`: Maximum tokens to generate
- `top_p::Float64`: Nucleus sampling threshold (0.0-1.0)
- `frequency_penalty::Float64`: Penalize frequent tokens (-2.0 to 2.0)
- `presence_penalty::Float64`: Penalize present tokens (-2.0 to 2.0)
- `stop::Union{String, Vector{String}}`: Stop sequences
- `n::Int`: Number of completions to generate
- `stream::Bool`: Enable streaming (handled automatically by streamcallback)
- `logprobs::Bool`: Include log probabilities
- `top_logprobs::Int`: Number of top log probabilities to return
- `seed::Int`: Random seed for deterministic sampling
- `response_format::Dict`: Structured output format (e.g., `Dict("type" => "json_object")`)

## AnthropicSchema
Anthropic-specific parameters:
- `max_tokens::Int`: Maximum tokens to generate (required, default 1000)
- `temperature::Float64`: Sampling temperature (0.0-1.0)
- `top_p::Float64`: Nucleus sampling (0.0-1.0)
- `top_k::Int`: Top-k sampling
- `stop_sequences::Vector{String}`: Stop sequences
- `cache::Symbol`: Prompt caching mode (`:system`, `:tools`, `:last`, `:all`, `:all_but_last`)
- `metadata::Dict`: Request metadata

## GeminiSchema
Google Gemini-specific parameters:
- `temperature::Float64`: Sampling temperature
- `top_p::Float64` / `topP::Float64`: Nucleus sampling
- `top_k::Int` / `topK::Int`: Top-k sampling
- `max_output_tokens::Int` / `maxOutputTokens::Int`: Maximum output tokens
- `presence_penalty::Float64` / `presencePenalty::Float64`: Presence penalty
- `frequency_penalty::Float64` / `frequencyPenalty::Float64`: Frequency penalty
- `response_mime_type::String` / `responseMimeType::String`: Output MIME type
- `response_schema::Dict` / `responseSchema::Dict`: Output schema
- `response_json_schema::Dict` / `responseJsonSchema::Dict`: JSON schema
- `stop_sequences::Vector{String}` / `stopSequences::Vector{String}`: Stop sequences
- `thinkingConfig::Dict`: Thinking/reasoning configuration
  - `thinkingLevel::Int`: Reasoning depth level
  - `thinkingBudget::Int`: Token budget for reasoning
  - `include_thoughts::Bool`: Include reasoning in response
- `candidateCount::Int`: Number of response candidates
- `seed::Int`: Random seed
- `responseLogprobs::Bool`: Include log probabilities
- `logprobs::Int`: Number of log probabilities

## ResponseSchema (OpenAI Response API)
For gpt-5 and o-series models:
- `max_completion_tokens::Int`: Maximum tokens in completion
- `reasoning_effort::String`: Reasoning effort level ("low", "medium", "high")
- `temperature::Float64`: Sampling temperature
- `top_p::Float64`: Nucleus sampling
- `modalities::Vector{String}`: Output modalities (e.g., ["text", "audio"])
- `audio::Dict`: Audio output configuration

# Examples
```julia
# OpenAI-compatible config
config = ModelConfig("openai:openai/gpt-5.1"; 
    temperature=0.7, 
    max_tokens=1000,
    top_p=0.9
)

# Anthropic with caching
config = ModelConfig("anthropic:anthropic/claude-sonnet-4.5";
    max_tokens=2000,
    temperature=0.8,
    cache=:all
)

# Gemini with thinking
config = ModelConfig("google-ai-studio:google/gemini-2.5-flash";
    temperature=0.7,
    maxOutputTokens=2000,
    thinkingConfig=Dict(
        :thinkingLevel => 2,
        :include_thoughts => true
    )
)

# Modify config later
config.kwargs = merge(config.kwargs, (temperature=0.9,))
```
"""
Base.@kwdef mutable struct ModelConfig
    slug::String  # provider:author/modelid format
    schema::Union{AbstractRequestSchema, Nothing} = nothing
    kwargs::NamedTuple = NamedTuple()
end

"""
    ModelConfig(slug::String; schema=nothing, kwargs...)

Create a ModelConfig with the given slug and optional parameters.

# Example
```julia
config = ModelConfig("openai:openai/gpt-5.1"; temperature=0.7, max_tokens=1000)
response = aigen("Hello", config)
```
"""
function ModelConfig(slug::String; schema::Union{AbstractRequestSchema, Nothing}=nothing, kwargs...)
    ModelConfig(slug=slug, schema=schema, kwargs=NamedTuple(kwargs))
end

"""
Extract slug and merge kwargs from ModelConfig with call-time kwargs.
Call-time kwargs take precedence over config kwargs.
"""
function extract_config(config::ModelConfig, call_kwargs)
    # Merge config kwargs with call kwargs (call kwargs take precedence)
    merged_kwargs = merge(NamedTuple(config.kwargs), call_kwargs)
    return config.slug, config.schema, merged_kwargs
end

"""
    list_schema_parameters(schema::Type{<:AbstractRequestSchema})
    list_schema_parameters(schema::AbstractRequestSchema)

List common parameters supported by a schema type.

# Example
```julia
list_schema_parameters(ChatCompletionSchema)
list_schema_parameters(AnthropicSchema)
list_schema_parameters(GeminiSchema)
```
"""
function list_schema_parameters(::Type{ChatCompletionSchema})
    return [
        :temperature => "Sampling temperature (0.0-2.0)",
        :max_tokens => "Maximum tokens to generate",
        :top_p => "Nucleus sampling threshold (0.0-1.0)",
        :frequency_penalty => "Penalize frequent tokens (-2.0 to 2.0)",
        :presence_penalty => "Penalize present tokens (-2.0 to 2.0)",
        :stop => "Stop sequences (String or Vector{String})",
        :n => "Number of completions to generate",
        :logprobs => "Include log probabilities",
        :top_logprobs => "Number of top log probabilities",
        :seed => "Random seed for deterministic sampling",
        :response_format => "Structured output format"
    ]
end

function list_schema_parameters(::Type{AnthropicSchema})
    return [
        :max_tokens => "Maximum tokens to generate (required, default 1000)",
        :temperature => "Sampling temperature (0.0-1.0)",
        :top_p => "Nucleus sampling (0.0-1.0)",
        :top_k => "Top-k sampling",
        :stop_sequences => "Stop sequences (Vector{String})",
        :cache => "Prompt caching mode (:system, :tools, :last, :all, :all_but_last)",
        :metadata => "Request metadata (Dict)"
    ]
end

function list_schema_parameters(::Type{GeminiSchema})
    return [
        :temperature => "Sampling temperature",
        :top_p => "Nucleus sampling (also accepts :topP)",
        :top_k => "Top-k sampling (also accepts :topK)",
        :max_output_tokens => "Maximum output tokens (also accepts :maxOutputTokens)",
        :presence_penalty => "Presence penalty (also accepts :presencePenalty)",
        :frequency_penalty => "Frequency penalty (also accepts :frequencyPenalty)",
        :response_mime_type => "Output MIME type (also accepts :responseMimeType)",
        :response_schema => "Output schema (also accepts :responseSchema)",
        :response_json_schema => "JSON schema (also accepts :responseJsonSchema)",
        :stop_sequences => "Stop sequences (also accepts :stopSequences)",
        :thinkingConfig => "Thinking configuration (Dict with :thinkingLevel, :thinkingBudget, :include_thoughts)",
        :candidateCount => "Number of response candidates",
        :seed => "Random seed",
        :responseLogprobs => "Include log probabilities",
        :logprobs => "Number of log probabilities"
    ]
end

function list_schema_parameters(::Type{ResponseSchema})
    return [
        :max_completion_tokens => "Maximum tokens in completion",
        :reasoning_effort => "Reasoning effort level (low, medium, high)",
        :temperature => "Sampling temperature",
        :top_p => "Nucleus sampling",
        :modalities => "Output modalities (Vector{String})",
        :audio => "Audio output configuration (Dict)"
    ]
end

# Convenience methods for instances
list_schema_parameters(schema::AbstractRequestSchema) = list_schema_parameters(typeof(schema))

"""
    list_config_parameters(config::ModelConfig)

List parameters currently set in a ModelConfig.

# Example
```julia
config = ModelConfig("openai:openai/gpt-5.1"; temperature=0.7, max_tokens=1000)
list_config_parameters(config)
```
"""
function list_config_parameters(config::ModelConfig)
    return collect(pairs(config.kwargs))
end

# ============================================================================
# Provider-specific Config Examples
# ============================================================================

"""
    GeminiConfig(model_id::String; kwargs...)

Convenience constructor for Gemini models with common parameter validation.

# Gemini-specific Parameters
- `temperature::Float64`: Sampling temperature (0.0-2.0)
- `topP::Float64`: Nucleus sampling (0.0-1.0)
- `topK::Int`: Top-k sampling
- `maxOutputTokens::Int`: Maximum output tokens
- `thinkingConfig::NamedTuple`: Thinking configuration with fields:
  - `thinkingLevel::String`: Reasoning depth ("low" or "high") - only for Pro models
  - `thinkingBudget::Int`: Token budget for reasoning - for non-Pro models
  - `include_thoughts::Bool`: Include reasoning in response

# Examples
```julia
# Basic Gemini config
config = GeminiConfig("google/gemini-2.5-flash"; 
    temperature=0.7,
    maxOutputTokens=2000
)

# Pro model with thinking level
config = GeminiConfig("google/gemini-2.5-pro";
    temperature=0.8,
    maxOutputTokens=3000,
    thinkingConfig=(
        thinkingLevel="high",
        include_thoughts=true
    )
)

# Non-Pro model with thinking budget
config = GeminiConfig("google/gemini-2.5-flash-thinking";
    thinkingConfig=(
        thinkingBudget=1000,
        include_thoughts=true
    )
)

response = aigen("Explain quantum entanglement", config)
```
"""
function GeminiConfig(model_id::String; kwargs...)
    # Validate Gemini-specific parameters
    validated_kwargs = Dict{Symbol, Any}()
    is_pro_model = occursin("pro", lowercase(model_id))
    
    for (k, v) in kwargs
        if k == :temperature
            (0.0 <= v <= 2.0) || @warn "temperature should be between 0.0 and 2.0, got $v"
            validated_kwargs[k] = v
        elseif k == :topP || k == :top_p
            (0.0 <= v <= 1.0) || @warn "topP should be between 0.0 and 1.0, got $v"
            validated_kwargs[:topP] = v
        elseif k == :topK || k == :top_k
            (v > 0) || @warn "topK should be positive, got $v"
            validated_kwargs[:topK] = v
        elseif k == :maxOutputTokens || k == :max_output_tokens
            (v > 0) || @warn "maxOutputTokens should be positive, got $v"
            validated_kwargs[:maxOutputTokens] = v
        elseif k == :thinkingConfig
            # Validate thinking config structure
            if v isa NamedTuple
                thinking_dict = Dict{Symbol, Any}()
                
                # Check thinkingLevel (only for Pro models)
                if haskey(v, :thinkingLevel)
                    level = v.thinkingLevel
                    is_pro_model || @warn "thinkingLevel is only supported for Pro models, use thinkingBudget instead for $model_id"
                    (level in ("low", "high", :low, :high)) || @warn "thinkingLevel should be 'low' or 'high', got $level"
                    thinking_dict[:thinkingLevel] = string(level)
                end
                
                # Check thinkingBudget (for non-Pro models)
                if haskey(v, :thinkingBudget)
                    budget = v.thinkingBudget
                    is_pro_model && @warn "thinkingBudget is not supported for Pro models, use thinkingLevel instead for $model_id"
                    (budget >= 0) || @warn "thinkingBudget should be non-negative, got $budget"
                    thinking_dict[:thinkingBudget] = budget
                end
                
                # include_thoughts is valid for all
                if haskey(v, :include_thoughts)
                    thinking_dict[:include_thoughts] = v.include_thoughts
                end
                
                validated_kwargs[:thinkingConfig] = thinking_dict
            elseif v isa AbstractDict
                validated_kwargs[:thinkingConfig] = v
            else
                error("thinkingConfig must be a NamedTuple or Dict, got $(typeof(v))")
            end
        else
            validated_kwargs[k] = v
        end
    end
    
    # Ensure model_id includes provider prefix
    full_slug = occursin(":", model_id) ? model_id : "google-ai-studio:$model_id"
    
    return ModelConfig(
        slug=full_slug,
        schema=GeminiSchema(),
        kwargs=NamedTuple(validated_kwargs)
    )
end