using JSON3

struct Pricing
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

struct ProviderEndpoint
    name::String
    model_name::String
    context_length::Union{Int, Nothing}
    pricing::Pricing
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