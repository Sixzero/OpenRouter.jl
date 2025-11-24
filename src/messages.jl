abstract type AbstractMessage end

Base.@kwdef struct UserMessage <: AbstractMessage
    content::AbstractString
    name::Union{Nothing, String} = nothing
    image_data::Union{Nothing, Vector{String}} = nothing  # base64 or data URLs
    extras::Union{Nothing, Dict{Symbol, Any}} = nothing
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
    extras::Union{Nothing, Dict{Symbol, Any}} = nothing
end

Base.@kwdef struct SystemMessage <: AbstractMessage
    content::AbstractString
    name::Union{Nothing, String} = nothing
    extras::Union{Nothing, Dict{Symbol, Any}} = nothing
end

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

    # Warn on consecutive same-type messages
    for i in 2:length(msgs)
        if typeof(msgs[i]) === typeof(msgs[i - 1])
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
        role = m isa UserMessage ? "user" :
               m isa AIMessage   ? "assistant" :
               m isa SystemMessage ? "system" :
               error("Unknown message type $(typeof(m)) for OpenAI schema")

        # Handle images for UserMessage
        if m isa UserMessage && m.image_data !== nothing && !isempty(m.image_data)
            content = Any[Dict("type" => "text", "text" => m.content)]
            for img in m.image_data
                push!(content, Dict("type" => "image_url", "image_url" => Dict("url" => img)))
            end
            msg_dict = Dict("role" => role, "content" => content)
        else
            msg_dict = Dict("role" => role, "content" => m.content)
        end

        if m.name !== nothing
            msg_dict["name"] = m.name
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
        # Skip SystemMessage - it should be handled separately in build_payload
        if m isa SystemMessage
            continue
        end
        
        role = m isa UserMessage ? "user" :
               m isa AIMessage ? "assistant" :
               error("Unknown message type $(typeof(m)) for Anthropic schema")

        # Handle images for UserMessage
        if m isa UserMessage && m.image_data !== nothing && !isempty(m.image_data)
            content = Any[Dict{String, Any}("type" => "text", "text" => m.content)]
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

        msg_dict = Dict("role" => role, "content" => content)

        push!(out, msg_dict)
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
        # Skip SystemMessage - handled separately
        if m isa SystemMessage
            continue
        end
        
        # Determine role
        role = m isa UserMessage ? "user" :
               m isa AIMessage ? "model" :
               error("Unknown message type $(typeof(m)) for Gemini schema")
        
        if m isa UserMessage && m.image_data !== nothing && !isempty(m.image_data)
            # Gemini multimodal format
            parts = Any[Dict("text" => m.content)]
            for img in m.image_data
                data_type, data = extract_image_attributes(img)
                # Gemini expects inline_data format
                push!(parts, Dict("inline_data" => Dict("mime_type" => data_type, "data" => data)))
            end
            push!(out, Dict("role" => role, "parts" => parts))
        elseif isa(m.content, AbstractString)
            push!(out, Dict("role" => role, "parts" => Any[Dict("text" => m.content)]))
        else
            push!(out, Dict("role" => role, "parts" => m.content))
        end
    end
    return out
end