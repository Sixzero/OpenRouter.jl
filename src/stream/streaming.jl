
using HTTP, JSON3

include("interface.jl")
include("shared_methods.jl")
include("http_stream_callback.jl")
include("http_stream_hooks.jl")
include("stream_chatcompletion.jl")
include("stream_anthropic.jl")
include("stream_gemini.jl")
include("stream_response.jl")

# ChatCompletionAnthropicSchema streaming forwards
acc_tokens(s::ChatCompletionAnthropicSchema, acc::TokenCounts, new::TokenCounts) = acc_tokens(_ccs, acc, new)
extract_ttft_ms(s::ChatCompletionAnthropicSchema, chunk::AbstractStreamChunk) = extract_ttft_ms(_ccs, chunk)
is_done(s::ChatCompletionAnthropicSchema, chunk::AbstractStreamChunk; kw...) = is_done(_ccs, chunk; kw...)
is_start(s::ChatCompletionAnthropicSchema, chunk::AbstractStreamChunk; kw...) = is_start(_ccs, chunk; kw...)
extract_content(s::ChatCompletionAnthropicSchema, chunk::AbstractStreamChunk; kw...) = extract_content(_ccs, chunk; kw...)
extract_reasoning_from_chunk(s::ChatCompletionAnthropicSchema, chunk::AbstractStreamChunk) = extract_reasoning_from_chunk(_ccs, chunk)
build_response_body(s::ChatCompletionAnthropicSchema, cb::AbstractLLMStream; kw...) = build_response_body(_ccs, cb; kw...)
extract_stop_sequence_from_chunk(s::ChatCompletionAnthropicSchema, chunk::AbstractStreamChunk) = extract_stop_sequence_from_chunk(_ccs, chunk)
extract_model_from_chunk(s::ChatCompletionAnthropicSchema, chunk::StreamChunk) = extract_model_from_chunk(_ccs, chunk)
