
# Convert to/from Dict format for backward compatibility
function to_dict(tokens::TokenCounts)
    dict = Dict{Symbol, Int}(
        :prompt_tokens => tokens.prompt_tokens,
        :completion_tokens => tokens.completion_tokens,
        :total_tokens => tokens.total_tokens
    )
    
    tokens.input_cache_read > 0 && (dict[:input_cache_read] = tokens.input_cache_read)
    tokens.input_cache_write > 0 && (dict[:input_cache_write] = tokens.input_cache_write)
    tokens.internal_reasoning > 0 && (dict[:internal_reasoning] = tokens.internal_reasoning)
    tokens.input_audio_cache > 0 && (dict[:input_audio_cache] = tokens.input_audio_cache)
    
    return dict
end

function from_dict(dict::Dict)
    return TokenCounts(
        prompt_tokens = get(dict, :prompt_tokens, 0),
        completion_tokens = get(dict, :completion_tokens, 0),
        total_tokens = get(dict, :total_tokens, 0),
        input_cache_read = get(dict, :input_cache_read, 0),
        input_cache_write = get(dict, :input_cache_write, 0),
        internal_reasoning = get(dict, :internal_reasoning, 0),
        input_audio_cache = get(dict, :input_audio_cache, 0)
    )
end

# Cost calculation using existing Pricing struct and parse_price function
parse_price(x) = x === nothing ? 0.0 :
                 x isa Real ? float(x) :
                 x isa AbstractString ? (tryparse(Float64, x) === nothing ? 0.0 : tryparse(Float64, x)) :
                 0.0

"""
Calculate cost based on pricing and token usage.
"""
function calculate_cost(pricing::Pricing, tokens::Union{Nothing,TokenCounts,Dict})
    tokens === nothing && return nothing
    
    # Convert Dict to TokenCounts if needed (backward compatibility)
    if tokens isa Dict
        tokens = from_dict(tokens)
    end

    total_cost = 0.0

    # Fields are non-overlapping, no subtraction needed
    total_cost += tokens.prompt_tokens * parse_price(pricing.prompt)           # cache misses at full price
    total_cost += tokens.input_cache_read * parse_price(pricing.input_cache_read)  # cache hits at cache price
    total_cost += tokens.input_cache_write * parse_price(pricing.input_cache_write)
    total_cost += tokens.completion_tokens * parse_price(pricing.completion)
    total_cost += tokens.internal_reasoning * parse_price(pricing.internal_reasoning)
    total_cost += tokens.input_audio_cache * parse_price(pricing.input_audio_cache)

    if pricing.discount !== nothing
        disc = parse_price(pricing.discount)
        disc > 0.0 && (total_cost *= (1.0 - disc))
    end

    return total_cost > 0.0 ? total_cost : nothing
end

"""
Calculate cost for a given endpoint and token usage.
Unwraps `.pricing`. Warns if cost cannot be determined (e.g. missing pricing).
"""
function calculate_cost(endpoint::ProviderEndpoint, tokens::Union{Nothing,TokenCounts,Dict}, verbose::Bool=false)
    if endpoint.pricing === nothing
        verbose && @warn "No pricing available on endpoint; cannot calculate cost." endpoint=endpoint tokens=tokens
        return nothing
    end

    cost = calculate_cost(endpoint.pricing, tokens)

    if cost === nothing
        # warn even in non verbose mode, this cannot go silent
        @warn "Pricing present but resulted in zero/undefined cost; check pricing fields and tokens." endpoint=endpoint tokens=tokens
    end

    return cost
end