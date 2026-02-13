# Changelog

## v1.3.0

- Add `ChatCompletionAnthropicSchema` for Anthropic models proxied through ChatCompletion format (fixes negative `prompt_tokens` when cache tokens use additive semantics)
- Add `image_data` support to `ToolMessage` for all providers
- Add `extras` field to `ToolMessage`
- Fix cross-provider tool result serialization
- Fix streaming tool calls for Anthropic/Gemini, add Gemini tool support
- Add image-only input support: skip empty text blocks in message serializers
- Assert non-empty `UserMessage` (require content or image_data)
- Add `claude-opus-4.6` to anthropic model transform
- Add AutoTag workflow for automatic version tagging on push

## v1.2.0

- Add `set_provider!` for intentional provider overwrites (`add_provider` now warns on overwrite)
- Fix precompile: skip network requests for echo providers

## v1.1.1

- Add endpoint override support and `image_data` extraction
- Add `convert_tool` and `extract_images` for `ResponseSchema`
- Image generation examples for Google and OpenAI
- SiliconFlow and DeepSeek providers tested and supported
- ChatCompletion reasoning stream support
- TTFT extraction and `RunInfo` improvements
- Ollama provider support
- `ModelConfig` introduction
- `ResponseSchema` (OpenAI Responses API) support
- Embedding endpoint, model mapping, `TokenCounts`, `AIMessage`, `HttpStreamHooks`
