abstract type AbstractMessage end

struct UserMessage <: AbstractMessage
    content::AbstractString
    name::Union{Nothing, String}
    image_data::Union{Nothing, Vector{String}}  # base64 or data URLs
    extras::Union{Nothing, Dict{Symbol, Any}}
    function UserMessage(; content::AbstractString="", name=nothing, image_data=nothing, extras=nothing)
        has_content = !isempty(content)
        has_image = image_data !== nothing && !isempty(image_data)
        @assert has_content || has_image "UserMessage must have non-empty content or image_data"
        new(content, name, image_data, extras)
    end
end

Base.@kwdef struct AIMessage <: AbstractMessage
    content::AbstractString
    name::Union{Nothing, String} = nothing
    image_data::Union{Nothing, Vector{String}} = nothing
    finish_reason::Union{Nothing, String} = nothing
    tokens::Union{Nothing, TokenCounts} = nothing
    elapsed::Float64 = -1.0
    cost::Union{Nothing, Float64} = nothing
    reasoning::Union{Nothing, String} = nothing  # Store reasoning/thinking content
    tool_calls::Union{Nothing, Vector{Dict{String, Any}}} = nothing  # Store tool calls
    extras::Union{Nothing, Dict{Symbol, Any}} = nothing
end

Base.@kwdef struct SystemMessage <: AbstractMessage
    content::AbstractString
    name::Union{Nothing, String} = nothing
    extras::Union{Nothing, Dict{Symbol, Any}} = nothing
end

Base.@kwdef struct ToolMessage <: AbstractMessage
    content::AbstractString
    tool_call_id::String
    name::Union{Nothing, String} = nothing
    image_data::Union{Nothing, Vector{String}} = nothing  # base64 or data URLs
end

"""Parse the arguments from a tool_call dict, handling both JSON string and already-parsed Dict."""
function get_arguments(tool_call::Dict)::Dict{String,Any}
    raw = tool_call["function"]["arguments"]
    raw isa AbstractString ? JSON3.read(raw, Dict{String,Any}) : Dict{String,Any}(raw)
end

"""Create a ToolMessage from a tool_call dict and string result."""
ToolMessage(tool_call::Dict, content::AbstractString; image_data=nothing) = ToolMessage(
    content = content,
    tool_call_id = tool_call["id"],
    name = get(tool_call["function"], "name", nothing),
    image_data = image_data
)

"""Create a ToolMessage by running `fn(args::Dict{String,Any})` with the tool_call's parsed arguments."""
ToolMessage(tool_call::Dict, fn::Function; image_data=nothing) = ToolMessage(tool_call, string(fn(get_arguments(tool_call))); image_data)

"""
Normalize `prompt` + `sys_msg` into a flat vector of AbstractMessage.

Accepted `prompt` forms:
- String                => one UserMessage
- AbstractMessage       => wrapped in a vector
- Vector of items       => each element may be
    - AbstractMessage
    - String            => treated as UserMessage
    - anything else     => treated as UserMessage with that content
"""
function normalize_messages(prompt, sys_msg)
    msgs = AbstractMessage[]
    
    if sys_msg !== nothing
        push!(msgs, SystemMessage(content=sys_msg))
    end

    if isa(prompt, AbstractString)
        push!(msgs, UserMessage(content=prompt))
    elseif prompt isa AbstractMessage
        push!(msgs, prompt)
    elseif isa(prompt, AbstractVector)
        for m in prompt
            if m isa AbstractMessage
                push!(msgs, m)
            elseif isa(m, AbstractString)
                push!(msgs, UserMessage(content=m))
            else
                push!(msgs, UserMessage(content=m))
            end
        end
    else
        push!(msgs, UserMessage(content=prompt))
    end

    # Warn on consecutive same-type messages (ToolMessages are expected consecutive for parallel tool calls)
    for i in 2:length(msgs)
        if typeof(msgs[i]) === typeof(msgs[i - 1]) && !(msgs[i] isa ToolMessage)
            @warn "Consecutive messages of the same type detected" index=i type=typeof(msgs[i])
        end
    end

    return msgs
end

# Helper to extract image attributes from data URL
function extract_image_attributes(img::AbstractString)
    if startswith(img, "data:")
        # Format: "data:image/jpeg;base64,<base64data>"
        parts = split(img, ",", limit=2)
        if length(parts) == 2
            header = parts[1]
            data = parts[2]
            # Extract media type from header
            media_match = match(r"data:([^;]+)", header)
            media_type = media_match !== nothing ? media_match.captures[1] : "image/jpeg"
            return media_type, data
        end
    end
    # Assume it's raw base64 data, default to jpeg
    return "image/jpeg", img
end

# === Converters for specific providers ===

# OpenAI-style chat: Dict("role"=>"user/assistant/system","content"=>...)
function to_openai_messages(msgs::Vector{AbstractMessage})
    out = Any[]
    for m in msgs
        # Handle ToolMessage separately (different structure)
        if m isa ToolMessage
            push!(out, Dict("role" => "tool", "tool_call_id" => m.tool_call_id, "content" => m.content))
            # OpenAI doesn't support images in tool results; inject as a following user message
            if m.image_data !== nothing && !isempty(m.image_data)
                img_parts = Any[]
                for img in m.image_data
                    push!(img_parts, Dict("type" => "image_url", "image_url" => Dict("url" => img)))
                end
                push!(out, Dict("role" => "user", "content" => img_parts))
            end
            continue
        end
        
        role = m isa UserMessage ? "user" :
               m isa AIMessage   ? "assistant" :
               m isa SystemMessage ? "system" :
               error("Unknown message type $(typeof(m)) for OpenAI schema")

        # Initialize message dict
        msg_dict = Dict{String, Any}("role" => role)
        
        # Handle images for UserMessage
        if m isa UserMessage && m.image_data !== nothing && !isempty(m.image_data)
            content = Any[]
            !isempty(m.content) && push!(content, Dict("type" => "text", "text" => m.content))
            for img in m.image_data
                push!(content, Dict("type" => "image_url", "image_url" => Dict("url" => img)))
            end
            msg_dict["content"] = content
        else
            msg_dict["content"] = m.content
        end

        if m.name !== nothing
            msg_dict["name"] = m.name
        end
        
        # Handle AIMessage extras (tool_calls, reasoning_content)
        if m isa AIMessage
            if m.tool_calls !== nothing
                msg_dict["tool_calls"] = m.tool_calls
                # Providers expect null content when only tool_calls are present
                isempty(m.content) && (msg_dict["content"] = nothing)
            end
            m.reasoning !== nothing && (msg_dict["reasoning_content"] = m.reasoning)
        end

        push!(out, msg_dict)
    end
    return out
end

# Anthropic-style: role + content=[{type="text", text=...}, ...]
to_anthropic_content(x) = isa(x, AbstractString) ? Any[Dict{String, Any}("type" => "text", "text" => x)] : x

function to_anthropic_messages(msgs::Vector{AbstractMessage}; cache::Union{Nothing,Symbol}=nothing)
    out = Any[]
    for m in msgs
        if m isa SystemMessage
            continue
        end

        # ToolMessage → tool_result block in a user message
        if m isa ToolMessage
            # Anthropic natively supports images in tool_result content arrays
            if m.image_data !== nothing && !isempty(m.image_data)
                tr_content = Any[]
                !isempty(m.content) && push!(tr_content, Dict{String,Any}("type" => "text", "text" => m.content))
                for img in m.image_data
                    data_type, data = extract_image_attributes(img)
                    push!(tr_content, Dict{String,Any}("type" => "image",
                        "source" => Dict{String,Any}("type" => "base64", "data" => data, "media_type" => data_type)))
                end
                tool_result_content = tr_content
            else
                tool_result_content = m.content
            end
            tool_result = Dict{String,Any}(
                "type" => "tool_result",
                "tool_use_id" => m.tool_call_id,
                "content" => tool_result_content
            )
            # Group consecutive ToolMessages into one user message (Anthropic requires alternation)
            if !isempty(out) && out[end]["role"] == "user" &&
               !isempty(out[end]["content"]) && out[end]["content"][end] isa Dict &&
               get(out[end]["content"][end], "type", "") == "tool_result"
                push!(out[end]["content"], tool_result)
            else
                push!(out, Dict("role" => "user", "content" => Any[tool_result]))
            end
            continue
        end

        role = m isa UserMessage ? "user" :
               m isa AIMessage ? "assistant" :
               error("Unknown message type $(typeof(m)) for Anthropic schema")

        # Handle images for UserMessage
        if m isa UserMessage && m.image_data !== nothing && !isempty(m.image_data)
            content = Any[]
            !isempty(m.content) && push!(content, Dict{String, Any}("type" => "text", "text" => m.content))
            for img in m.image_data
                data_type, data = extract_image_attributes(img)
                @assert data_type in ["image/jpeg", "image/png", "image/gif", "image/webp"] "Unsupported image type: $data_type"
                push!(content, Dict("type" => "image",
                    "source" => Dict("type" => "base64",
                        "data" => data,
                        "media_type" => data_type)))
            end
        else
            content = to_anthropic_content(m.content)
        end

        # AIMessage with tool_calls → append tool_use blocks
        if m isa AIMessage && m.tool_calls !== nothing
            # Remove empty text block — Anthropic rejects empty text alongside tool_use
            if !isempty(content) && content[1] isa Dict &&
               get(content[1], "type", "") == "text" && isempty(get(content[1], "text", ""))
                popfirst!(content)
            end
            for tc in m.tool_calls
                fn = tc["function"]
                args = get_arguments(tc)
                push!(content, Dict{String,Any}(
                    "type" => "tool_use",
                    "id" => tc["id"],
                    "name" => fn["name"],
                    "input" => args
                ))
            end
        end

        push!(out, Dict("role" => role, "content" => content))
    end

    # Apply Anthropic prompt caching markers if requested
    if cache !== nothing
        @assert cache in (:system, :tools, :last, :all, :all_but_last) "Unsupported cache mode: $cache"
        # Only user messages are considered here; system is handled in build_payload
        user_msg_counter = 0
        for i in reverse(eachindex(out))
            out[i]["role"] == "user" || continue
            haskey(out[i], "content") && !isempty(out[i]["content"]) || continue
            last_block = out[i]["content"][end]
            cache === :last && user_msg_counter == 0 &&
                (last_block["cache_control"] = Dict("type" => "ephemeral"))
            cache === :all && user_msg_counter < 2 &&
                (last_block["cache_control"] = Dict("type" => "ephemeral"))
            cache === :all_but_last && user_msg_counter == 1 &&
                (last_block["cache_control"] = Dict("type" => "ephemeral"))
            user_msg_counter += 1
        end
    end

    return out
end

# Gemini: we treat all as "contents" items; system_instruction is separate.
function to_gemini_contents(msgs::Vector{AbstractMessage})
    out = Any[]
    for m in msgs
        if m isa SystemMessage
            continue
        end

        # ToolMessage → functionResponse part in a user message
        if m isa ToolMessage
            part = Dict{String,Any}(
                "functionResponse" => Dict{String,Any}(
                    "name" => something(m.name, m.tool_call_id),
                    "response" => Dict{String,Any}("content" => m.content)
                )
            )
            # Build image parts if present
            img_parts = Any[]
            if m.image_data !== nothing && !isempty(m.image_data)
                for img in m.image_data
                    data_type, data = extract_image_attributes(img)
                    push!(img_parts, Dict{String,Any}("inline_data" => Dict{String,Any}("mime_type" => data_type, "data" => data)))
                end
            end
            # Group consecutive ToolMessages into one user message
            if !isempty(out) && out[end]["role"] == "user" &&
               !isempty(out[end]["parts"]) && haskey(out[end]["parts"][end], "functionResponse")
                push!(out[end]["parts"], part)
                append!(out[end]["parts"], img_parts)
            else
                push!(out, Dict("role" => "user", "parts" => Any[part; img_parts...]))
            end
            continue
        end

        role = m isa UserMessage ? "user" :
               m isa AIMessage ? "model" :
               error("Unknown message type $(typeof(m)) for Gemini schema")

        if m isa UserMessage && m.image_data !== nothing && !isempty(m.image_data)
            parts = Any[]
            !isempty(m.content) && push!(parts, Dict("text" => m.content))
            for img in m.image_data
                data_type, data = extract_image_attributes(img)
                push!(parts, Dict("inline_data" => Dict("mime_type" => data_type, "data" => data)))
            end
            push!(out, Dict("role" => role, "parts" => parts))
        elseif isa(m.content, AbstractString)
            parts = isempty(m.content) ? Any[] : Any[Dict("text" => m.content)]
            # AIMessage with tool_calls → append functionCall parts
            if m isa AIMessage && m.tool_calls !== nothing
                for tc in m.tool_calls
                    fn = tc["function"]
                    args = get_arguments(tc)
                    push!(parts, Dict{String,Any}("functionCall" => Dict{String,Any}("name" => fn["name"], "args" => args)))
                end
            end
            push!(out, Dict("role" => role, "parts" => parts))
        else
            push!(out, Dict("role" => role, "parts" => m.content))
        end
    end
    return out
end