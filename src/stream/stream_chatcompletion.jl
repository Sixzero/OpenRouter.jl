# Custom methods for ChatCompletion streaming (OpenAI-compatible)

"""
    is_done(schema::ChatCompletionSchema, chunk::AbstractStreamChunk; kwargs...)

Check if streaming is done for ChatCompletion format.
Checks for finish_reason in choices or [DONE] marker.
"""
@inline function is_done(schema::ChatCompletionSchema, chunk::AbstractStreamChunk; kwargs...)
    # Check for [DONE] marker
    chunk.data == "[DONE]" && return true

    # Check for finish_reason in choices
    if !isnothing(chunk.json)
        choices = get(chunk.json, :choices, [])
        if !isempty(choices)
            first_choice = choices[1]
            finish_reason = get(first_choice, :finish_reason, nothing)
            # Any non-nothing finish_reason means we're done
            return !isnothing(finish_reason)
        end
    end

    return false
end

"""
    is_start(schema::ChatCompletionSchema, chunk::AbstractStreamChunk; kwargs...)

Check if streaming has started for ChatCompletion format.
"""
@inline function is_start(schema::ChatCompletionSchema, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        # OpenAI-compatible ChatCompletion streams typically send a role in the first delta
        choices = get(chunk.json, :choices, [])
        isempty(choices) && return false
        first_choice = choices[1]
        delta = get(first_choice, :delta, Dict())
        return haskey(delta, :role)
    end
    false
end

"""
    extract_content(schema::ChatCompletionSchema, chunk::AbstractStreamChunk; kwargs...)

Extract content from ChatCompletion chunk.
"""
@inline function extract_content(schema::ChatCompletionSchema, chunk::AbstractStreamChunk; kwargs...)
    if !isnothing(chunk.json)
        choices = get(chunk.json, :choices, [])
        first_choice = get(choices, 1, Dict())
        delta = get(first_choice, :delta, Dict())
        return get(delta, :content, nothing)
    end
    return nothing
end

"""
    build_response_body(schema::ChatCompletionSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)

Build response body from chunks to mimic standard ChatCompletion API response.

Note: Limited functionality. Does NOT support tool use, refusals, logprobs.
"""
function build_response_body(schema::ChatCompletionSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)
    isempty(cb.chunks) && return nothing
    
    response = nothing
    usage = nothing
    choices_output = Dict{Int, Dict{Symbol, Any}}()
    
    for chunk in cb.chunks
        isnothing(chunk.json) && continue
        !haskey(chunk.json, :choices) && continue
        
        if isnothing(response)
            response = chunk.json |> copy
        end
        
        if isnothing(usage)
            usage_values = get(chunk.json, :usage, nothing)
            !isnothing(usage_values) && (usage = usage_values |> copy)
        end
        
        for choice in chunk.json.choices
            index = get(choice, :index, nothing)
            isnothing(index) && continue
            
            if !haskey(choices_output, index)
                choices_output[index] = Dict{Symbol, Any}(:index => index)
            end
            
            index_dict = choices_output[index]
            finish_reason = get(choice, :finish_reason, nothing)
            !isnothing(finish_reason) && (index_dict[:finish_reason] = finish_reason)
            
            choice_delta = get(choice, :delta, Dict{Symbol, Any}())
            message_dict = get(index_dict, :message, Dict{Symbol, Any}(:content => ""))
            
            role = get(choice_delta, :role, nothing)
            !isnothing(role) && (message_dict[:role] = role)
            
            content = get(choice_delta, :content, nothing)
            !isnothing(content) && (message_dict[:content] *= content)
            
            index_dict[:message] = message_dict
        end
    end
    
    if !isnothing(response)
        choices = [choices_output[index] for index in sort(collect(keys(choices_output)))]
        response[:choices] = choices
        response[:object] = "chat.completion"
        !isnothing(usage) && (response[:usage] = usage)
    end
    
    return response
end