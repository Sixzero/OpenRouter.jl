# ## Types
"""
    AbstractStreamChunk

Abstract type for stream chunks.

Must have fields:
- `event`: The event name
- `data`: The data chunk  
- `json`: The JSON object or `nothing` if chunk doesn't contain JSON
"""
abstract type AbstractStreamChunk end

"""
    AbstractLLMStream

Abstract type for LLM stream callbacks.

Must have fields:
- `out`: Output stream (e.g., `stdout` or pipe)
- `schema`: Request schema determining API format
- `chunks`: List of received `AbstractStreamChunk` chunks
- `verbose`: Whether to print verbose information
- `kwargs`: Custom keyword arguments
"""
abstract type AbstractLLMStream end