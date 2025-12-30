# Custom methods for ChatCompletion streaming (OpenAI-compatible)

"""
Accumulate tokens for ChatCompletion schema.
Detects cumulative counts (DeepSeek style) vs delta counts and handles accordingly.
"""
function acc_tokens(schema::ChatCompletionSchema, accumulator::TokenCounts, new_tokens::TokenCounts)
    # Detect cumulative counts: if new completion_tokens >= old, it's cumulative (replace)
    if new_tokens.completion_tokens >= accumulator.completion_tokens
        return new_tokens
    end
    # Otherwise delta-based (add)
    return accumulator + new_tokens
end

"""
Extract ttft_ms from sla_metrics (DeepSeek/OpenRouter style).
"""
function extract_ttft_ms(schema::ChatCompletionSchema, chunk::AbstractStreamChunk)
    isnothing(chunk.json) && return nothing
    sla_metrics = get(chunk.json, :sla_metrics, nothing)
    isnothing(sla_metrics) && return nothing
    return get(sla_metrics, :ttft_ms, nothing)
end

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
    extract_reasoning_from_chunk(schema::ChatCompletionSchema, chunk::AbstractStreamChunk)

Extract reasoning_content from ChatCompletion chunk (DeepSeek style).
"""
@inline function extract_reasoning_from_chunk(schema::ChatCompletionSchema, chunk::AbstractStreamChunk)
    if !isnothing(chunk.json)
        choices = get(chunk.json, :choices, [])
        first_choice = get(choices, 1, Dict())
        delta = get(first_choice, :delta, Dict())
        return get(delta, :reasoning_content, nothing)
    end
    return nothing
end

"""
    build_response_body(schema::ChatCompletionSchema, cb::AbstractLLMStream; verbose::Bool = false, kwargs...)

Build response body from chunks to mimic standard ChatCompletion API response.

Note: Limited functionality. Does NOT support refusals, logprobs.
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
        
        # Always take latest usage (handles cumulative providers like DeepSeek)
        usage_values = get(chunk.json, :usage, nothing)
        !isnothing(usage_values) && (usage = usage_values)
        
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
            
            # Accumulate reasoning_content (DeepSeek style)
            reasoning = get(choice_delta, :reasoning_content, nothing)
            !isnothing(reasoning) && (message_dict[:reasoning_content] = get(message_dict, :reasoning_content, "") * reasoning)

            # Accumulate tool_calls (OpenAI streaming format: choices[].delta.tool_calls)
            delta_tool_calls = get(choice_delta, :tool_calls, nothing)
            if !isnothing(delta_tool_calls)
                # Keep as Dict(index=>call) while accumulating; convert to Vector at end.
                tool_calls_acc = get(message_dict, :tool_calls, Dict{Int, Dict{Symbol, Any}}())
                for tc in delta_tool_calls
                    tc_index = get(tc, :index, 0)
                    tc_dict = get!(tool_calls_acc, tc_index, Dict{Symbol, Any}())

                    tc_id = get(tc, :id, nothing)
                    !isnothing(tc_id) && (tc_dict[:id] = tc_id)

                    tc_type = get(tc, :type, nothing)
                    !isnothing(tc_type) && (tc_dict[:type] = tc_type)

                    tc_func = get(tc, :function, nothing)
                    if !isnothing(tc_func)
                        func_dict = get(tc_dict, :function, Dict{Symbol, Any}())
                        func_name = get(tc_func, :name, nothing)
                        !isnothing(func_name) && (func_dict[:name] = func_name)
                        func_args = get(tc_func, :arguments, nothing)
                        !isnothing(func_args) && (func_dict[:arguments] = get(func_dict, :arguments, "") * func_args)
                        tc_dict[:function] = func_dict
                    end
                end
                message_dict[:tool_calls] = tool_calls_acc
            end
            
            index_dict[:message] = message_dict
        end
    end
    
    if !isnothing(response)
        choices = [choices_output[index] for index in sort(collect(keys(choices_output)))]
        # Convert tool_calls accumulator dict to a stable ordered vector
        for choice in choices
            msg = get(choice, :message, nothing)
            isnothing(msg) && continue
            tc_acc = get(msg, :tool_calls, nothing)
            tc_acc isa AbstractDict || continue
            msg[:tool_calls] = [tc_acc[i] for i in sort(collect(keys(tc_acc)))]
        end
        response[:choices] = choices
        response[:object] = "chat.completion"
        !isnothing(usage) && (response[:usage] = usage)
    end
    
    return response
end