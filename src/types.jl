using JSON3

"""
    Tool

Schema-agnostic tool definition. Automatically converted to the right format
based on the API schema (OpenAI vs Anthropic vs Gemini).

# Example
```julia
tool = Tool(
    name = "create_file",
    description = "Create a file with the given content",
    parameters = Dict(
        "type" => "object",
        "properties" => Dict(
            "path" => Dict("type" => "string", "description" => "File path"),
            "content" => Dict("type" => "string", "description" => "File content")
        ),
        "required" => ["path", "content"]
    )
)
```
"""
Base.@kwdef struct Tool
    name::String
    description::String
    parameters::Dict{String, Any}
end

Base.@kwdef struct Pricing
    prompt::Union{String, Nothing}
    completion::Union{String, Nothing}
    request::Union{String, Nothing}
    image::Union{String, Nothing}
    web_search::Union{String, Nothing}
    internal_reasoning::Union{String, Nothing}
    image_output::Union{String, Nothing}
    audio::Union{String, Nothing}
    input_audio_cache::Union{String, Nothing}
    input_cache_read::Union{String, Nothing}
    input_cache_write::Union{String, Nothing}
    discount::Union{Float64, Nothing}
end

struct Architecture
    modality::Union{String, Nothing}
    input_modalities::Union{Vector{String}, Nothing}
    output_modalities::Union{Vector{String}, Nothing}
    tokenizer::Union{String, Nothing}
    instruct_type::Union{String, Nothing}
end

struct OpenRouterModel
    id::String
    name::String
    description::Union{String, Nothing}
    context_length::Union{Int, Nothing}
    pricing::Union{Pricing, Nothing}
    architecture::Union{Architecture, Nothing}
    created::Union{Int, Nothing}
end

@kwdef mutable struct ProviderEndpoint
    name::String
    model_name::String
    context_length::Union{Int, Nothing}
    pricing::Union{Pricing, Nothing}
    provider_name::String
    tag::Union{String, Nothing}
    quantization::Union{String, Nothing}
    max_completion_tokens::Union{Int, Nothing}
    max_prompt_tokens::Union{Int, Nothing}
    supported_parameters::Union{Vector{String}, Nothing}
    uptime_last_30m::Union{Float64, Nothing}
    supports_implicit_caching::Union{Bool, Nothing}
    status::Union{Int, Nothing}
end

struct ModelProviders
    id::String
    name::String
    created::Union{Int, Nothing}
    description::Union{String, Nothing}
    architecture::Union{Architecture, Nothing}
    endpoints::Vector{ProviderEndpoint}
end

# New structs from patch
struct TopProvider
    is_moderated::Union{Bool, Nothing}
    context_length::Union{Int, Nothing}
    max_completion_tokens::Union{Int, Nothing}
end

struct OpenRouterEmbeddingModel
    id::String
    canonical_slug::Union{String, Nothing}
    name::String
    created::Union{Int, Nothing}
    pricing::Union{Pricing, Nothing}
    context_length::Union{Int, Nothing}
    architecture::Union{Architecture, Nothing}
    top_provider::Union{TopProvider, Nothing}
    per_request_limits::Union{Nothing, Nothing}  # Always null in API
    supported_parameters::Union{Vector{String}, Nothing}
    default_parameters::Union{Nothing, Nothing}  # Always null in API
    description::Union{String, Nothing}
end


"""
Universal token counting and cost calculation utilities for OpenRouter.jl

This module provides standardized token counting and cost calculation across all schemas and providers.
"""

"""
    TokenCounts

Universal token counting struct. Fields are NON-OVERLAPPING for correct cost calculation.

# Fields (non-overlapping)
- `prompt_tokens::Int`: Cache misses - input tokens NOT served from cache (charged at full price)
- `input_cache_read::Int`: Cache hits - input tokens served from cache (charged at cache price)
- `completion_tokens::Int`: Output tokens
- `total_tokens::Int`: Sum of all input + output tokens
- `input_cache_write::Int`: Tokens written to cache (Anthropic)
- `internal_reasoning::Int`: Reasoning/thinking tokens (Gemini, DeepSeek R1)
- `input_audio_cache::Int`: Audio tokens cached

# Cost calculation
Total input = prompt_tokens + input_cache_read (no double counting)
Cost = prompt_tokens × full_price + input_cache_read × cache_price
"""
Base.@kwdef struct TokenCounts
    prompt_tokens::Int = 0          # cache misses (non-cached input)
    completion_tokens::Int = 0
    total_tokens::Int = 0
    input_cache_read::Int = 0       # cache hits
    input_cache_write::Int = 0
    internal_reasoning::Int = 0
    input_audio_cache::Int = 0
end

# Arithmetic operations for TokenCounts
Base.:+(a::TokenCounts, b::TokenCounts) = TokenCounts(
    prompt_tokens = a.prompt_tokens + b.prompt_tokens,
    completion_tokens = a.completion_tokens + b.completion_tokens,
    total_tokens = a.total_tokens + b.total_tokens,
    input_cache_read = a.input_cache_read + b.input_cache_read,
    input_cache_write = a.input_cache_write + b.input_cache_write,
    internal_reasoning = a.internal_reasoning + b.internal_reasoning,
    input_audio_cache = a.input_audio_cache + b.input_audio_cache
)

# Custom display for TokenCounts - only show non-zero fields
function Base.show(io::IO, tc::TokenCounts)
    fields = []
    for field in fieldnames(TokenCounts)
        value = getfield(tc, field)
        if value != 0
            push!(fields, "$field=$value")
        end
    end

    if isempty(fields)
        print(io, "TokenCounts(all zeros)")
    else
        print(io, "TokenCounts(", join(fields, ", "), ")")
    end
end

"""
    parse_models(json_str::String)::Vector{OpenRouterModel}

Parse OpenRouter models JSON response into Julia structs.
"""
function parse_models(json_str::String)::Vector{OpenRouterModel}
    data = JSON3.read(json_str)
    
    models = OpenRouterModel[]
    for model_data in data.data
        pricing = haskey(model_data, :pricing) ? 
            Pricing(
                get(model_data.pricing, :prompt, nothing),
                get(model_data.pricing, :completion, nothing),
                get(model_data.pricing, :request, nothing),
                get(model_data.pricing, :image, nothing),
                get(model_data.pricing, :web_search, nothing),
                get(model_data.pricing, :internal_reasoning, nothing),
                get(model_data.pricing, :image_output, nothing),
                get(model_data.pricing, :audio, nothing),
                get(model_data.pricing, :input_audio_cache, nothing),
                get(model_data.pricing, :input_cache_read, nothing),
                get(model_data.pricing, :input_cache_write, nothing),
                get(model_data.pricing, :discount, nothing)
            ) : nothing
            
        architecture = haskey(model_data, :architecture) ?
            Architecture(
                get(model_data.architecture, :modality, nothing),
                get(model_data.architecture, :input_modalities, nothing),
                get(model_data.architecture, :output_modalities, nothing),
                get(model_data.architecture, :tokenizer, nothing),
                get(model_data.architecture, :instruct_type, nothing)
            ) : nothing
            
        push!(models, OpenRouterModel(
            model_data.id,
            model_data.name,
            get(model_data, :description, nothing),
            get(model_data, :context_length, nothing),
            pricing,
            architecture,
            get(model_data, :created, nothing)
        ))
    end
    
    return models
end

"""
    parse_endpoints(json_str::String)::ModelProviders

Parse model providers JSON response into Julia struct.
"""
function parse_endpoints(json_str::String)::ModelProviders
    data = JSON3.read(json_str)
    model_data = data.data
    
    architecture = haskey(model_data, :architecture) ?
        Architecture(
            get(model_data.architecture, :modality, nothing),
            get(model_data.architecture, :input_modalities, nothing),
            get(model_data.architecture, :output_modalities, nothing),
            get(model_data.architecture, :tokenizer, nothing),
            get(model_data.architecture, :instruct_type, nothing)
        ) : nothing
    
    endpoints = ProviderEndpoint[]
    for endpoint_data in model_data.endpoints
        pricing = Pricing(
            get(endpoint_data.pricing, :prompt, nothing),
            get(endpoint_data.pricing, :completion, nothing),
            get(endpoint_data.pricing, :request, nothing),
            get(endpoint_data.pricing, :image, nothing),
            get(endpoint_data.pricing, :web_search, nothing),
            get(endpoint_data.pricing, :internal_reasoning, nothing),
            get(endpoint_data.pricing, :image_output, nothing),
            get(endpoint_data.pricing, :audio, nothing),
            get(endpoint_data.pricing, :input_audio_cache, nothing),
            get(endpoint_data.pricing, :input_cache_read, nothing),
            get(endpoint_data.pricing, :input_cache_write, nothing),
            get(endpoint_data.pricing, :discount, nothing)
        )
        
        push!(endpoints, ProviderEndpoint(
            endpoint_data.name,
            endpoint_data.model_name,
            get(endpoint_data, :context_length, nothing),
            pricing,
            endpoint_data.provider_name,
            get(endpoint_data, :tag, nothing),
            get(endpoint_data, :quantization, nothing),
            get(endpoint_data, :max_completion_tokens, nothing),
            get(endpoint_data, :max_prompt_tokens, nothing),
            get(endpoint_data, :supported_parameters, nothing),
            get(endpoint_data, :uptime_last_30m, nothing),
            get(endpoint_data, :supports_implicit_caching, nothing),
            get(endpoint_data, :status, nothing)
        ))
    end
    
    return ModelProviders(
        model_data.id,
        model_data.name,
        get(model_data, :created, nothing),
        get(model_data, :description, nothing),
        architecture,
        endpoints
    )
end

"""
    parse_embedding_models(json_str::String)::Vector{OpenRouterEmbeddingModel}

Parse OpenRouter embedding models JSON response into Julia structs.
"""
function parse_embedding_models(json_str::String)::Vector{OpenRouterEmbeddingModel}
    data = JSON3.read(json_str)
    
    models = OpenRouterEmbeddingModel[]
    for model_data in data.data
        pricing = haskey(model_data, :pricing) ? 
            Pricing(
                get(model_data.pricing, :prompt, nothing),
                get(model_data.pricing, :completion, nothing),
                get(model_data.pricing, :request, nothing),
                get(model_data.pricing, :image, nothing),
                get(model_data.pricing, :web_search, nothing),
                get(model_data.pricing, :internal_reasoning, nothing),
                get(model_data.pricing, :image_output, nothing),
                get(model_data.pricing, :audio, nothing),
                get(model_data.pricing, :input_audio_cache, nothing),
                get(model_data.pricing, :input_cache_read, nothing),
                get(model_data.pricing, :input_cache_write, nothing),
                get(model_data.pricing, :discount, nothing)
            ) : nothing
            
        architecture = haskey(model_data, :architecture) ?
            Architecture(
                get(model_data.architecture, :modality, nothing),
                get(model_data.architecture, :input_modalities, nothing),
                get(model_data.architecture, :output_modalities, nothing),
                get(model_data.architecture, :tokenizer, nothing),
                get(model_data.architecture, :instruct_type, nothing)
            ) : nothing

        top_provider = haskey(model_data, :top_provider) ?
            TopProvider(
                get(model_data.top_provider, :is_moderated, nothing),
                get(model_data.top_provider, :context_length, nothing),
                get(model_data.top_provider, :max_completion_tokens, nothing)
            ) : nothing
            
        push!(models, OpenRouterEmbeddingModel(
            model_data.id,
            get(model_data, :canonical_slug, nothing),
            model_data.name,
            get(model_data, :created, nothing),
            pricing,
            get(model_data, :context_length, nothing),
            architecture,
            top_provider,
            nothing,  # per_request_limits always null
            get(model_data, :supported_parameters, nothing),
            nothing,  # default_parameters always null
            get(model_data, :description, nothing)
        ))
    end
    
    return models
end