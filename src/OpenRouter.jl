module OpenRouter

include("types.jl")
include("stream_interface.jl")
include("api.jl")
include("storage.jl")
include("model_mapping.jl")
include("messages.jl")
include("schemas.jl")
include("schemas/response_schema.jl")
include("model_config.jl")
include("costs_tokens.jl")
include("providers.jl")
include("providers_native.jl")
include("aigen.jl")

# https streaming example.
include("stream/streaming.jl")

# Echo server for testing/precompilation
include("echo_server.jl")

# Precompilation workload
include("precompile.jl")

# Core API functions
export list_models, list_models_raw, list_providers, list_providers_raw
export get_model, list_cached_models, search_models
export aigen, aigen_raw
export list_provider_endpoints
export list_native_models, list_endpoints

# Types
export OpenRouterModel, Pricing, Architecture, ProviderEndpoint, ModelProviders, CachedModel, ModelCache
export ModelConfig, GeminiConfig

# Streaming types
export HttpStreamCallback, StreamChunk, HttpStreamHooks

# Note: add_provider, remove_provider, add_model, remove_model are NOT exported
# They are internal functionality that users can import explicitly if needed:
# using OpenRouter: add_provider, add_model

end
