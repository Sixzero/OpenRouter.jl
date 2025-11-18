using HTTP
using JSON3

"""
    aigen(prompt::String, provider_model::String; 
          api_key::Union{String, Nothing} = nothing,
          kwargs...)

Generate text using a specific provider and model.

# Arguments
- `prompt::String`: The input prompt
- `provider_model::String`: Format "Provider:model/slug" (e.g., "Together:moonshotai/kimi-k2-thinking")

# Keyword Arguments
- `api_key::Union{String, Nothing}`: Provider-specific API key (auto-detected from env if not provided)
- `kwargs...`: Additional API parameters

# Returns
- `String`: Generated text response

# Example
```julia
response = aigen("Write a haiku about Julia programming", "Together:moonshotai/kimi-k2-thinking")
println(response)
```
"""
function aigen(prompt::String, provider_model::String; 
               api_key::Union{String, Nothing} = nothing,
               kwargs...)
    
    provider_name, model_id = parse_provider_model(provider_model)
    @show provider_name
    @show model_id
    
    # Get provider info
    provider_info = get_provider_info(lowercase(provider_name))
    @show provider_info
    provider_info === nothing && throw(ArgumentError("Unknown provider: $provider_name"))
    
    # Get API key
    if api_key === nothing
        if provider_info.api_key_env_var !== nothing
            api_key = get(ENV, provider_info.api_key_env_var, "")
            isempty(api_key) && throw(ArgumentError("API key not found in environment variable $(provider_info.api_key_env_var)"))
        else
            throw(ArgumentError("No API key provided and no standard env var for provider $provider_name"))
        end
    end
    
    # Build request payload
    payload = Dict{String, Any}(
        "model" => model_id,
        "messages" => [Dict("role" => "user", "content" => prompt)]
    )
    
    # Add any additional kwargs
    for (k, v) in kwargs
        payload[string(k)] = v
    end
    
    # Build headers
    auth_header = get_provider_auth_header(lowercase(provider_name), api_key)
    auth_header === nothing && throw(ArgumentError("Failed to build auth header for provider $provider_name"))
    
    headers = [
        auth_header,
        "Content-Type" => "application/json"
    ]
    
    # Add default headers for this provider
    for (k, v) in provider_info.default_headers
        push!(headers, k => v)
    end
    
    # Build URL
    url = "$(provider_info.base_url)/chat/completions"
    
    try
        response = HTTP.post(url, headers, JSON3.write(payload))
        
        if response.status != 200
            error("API request failed with status $(response.status): $(String(response.body))")
        end
        
        result = JSON3.read(response.body)
        
        # Extract the generated text
        if haskey(result, :choices) && length(result.choices) > 0
            choice = result.choices[1]
            if haskey(choice, :message) && haskey(choice.message, :content)
                return choice.message.content
            end
        end
        
        error("Unexpected response format from API")
        
    catch e
        if e isa HTTP.ExceptionRequest.StatusError
            error("API request failed: $(e.response.status) - $(String(e.response.body))")
        else
            rethrow(e)
        end
    end
end

"""
    parse_provider_model(provider_model::String)::Tuple{String, String}

Parse "Provider:model/slug" format into provider name and model ID.
"""
function parse_provider_model(provider_model::String)::Tuple{String, String}
    parts = split(provider_model, ":", limit=2)
    length(parts) != 2 && throw(ArgumentError("Invalid format. Use 'Provider:model/slug'"))
    
    provider_name = strip(parts[1])
    model_id = strip(parts[2])
    
    isempty(provider_name) && throw(ArgumentError("Provider name cannot be empty"))
    isempty(model_id) && throw(ArgumentError("Model ID cannot be empty"))
    
    return provider_name, model_id
end

"""
    aigen_raw(prompt::String, provider_model::String; kwargs...)

Generate text and return the raw API response as a dictionary.
Useful for accessing usage statistics, multiple choices, etc.
"""
function aigen_raw(prompt::String, provider_model::String; 
                   api_key::Union{String, Nothing} = nothing,
                   kwargs...)
    
    provider_name, model_id = parse_provider_model(provider_model)
    
    # Get provider info
    provider_info = get_provider_info(lowercase(provider_name))
    provider_info === nothing && throw(ArgumentError("Unknown provider: $provider_name"))
    
    # Get API key
    if api_key === nothing
        if provider_info.api_key_env_var !== nothing
            api_key = get(ENV, provider_info.api_key_env_var, "")
            isempty(api_key) && throw(ArgumentError("API key not found in environment variable $(provider_info.api_key_env_var)"))
        else
            throw(ArgumentError("No API key provided and no standard env var for provider $provider_name"))
        end
    end
    
    # Build request payload
    payload = Dict{String, Any}(
        "model" => model_id,
        "messages" => [Dict("role" => "user", "content" => prompt)]
    )
    
    # Add any additional kwargs
    for (k, v) in kwargs
        payload[string(k)] = v
    end
    
    # Build headers
    auth_header = get_provider_auth_header(lowercase(provider_name), api_key)
    auth_header === nothing && throw(ArgumentError("Failed to build auth header for provider $provider_name"))
    
    headers = [
        auth_header,
        "Content-Type" => "application/json"
    ]
    
    # Add default headers for this provider
    for (k, v) in provider_info.default_headers
        push!(headers, k => v)
    end
    
    # Build URL
    url = "$(provider_info.base_url)/chat/completions"
    
    try
        response = HTTP.post(url, headers, JSON3.write(payload))
        
        if response.status != 200
            error("API request failed with status $(response.status): $(String(response.body))")
        end
        
        return JSON3.read(response.body)
        
    catch e
        if e isa HTTP.ExceptionRequest.StatusError
            error("API request failed: $(e.response.status) - $(String(e.response.body))")
        else
            rethrow(e)
        end
    end
end