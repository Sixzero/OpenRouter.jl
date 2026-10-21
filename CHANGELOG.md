# Changelog

## Unreleased

- `HttpStreamHooks.provenance`: per-call audit trail filled before `on_meta_ai` — API `host`, provider request-id headers (`x-request-id`, `x-amzn-requestid`, `apim-request-id`, `x-generation-id`, `x-ms-region`, …), `responseId`, `servedModel`, OpenRouter `upstream` provider
- Add `openrouter_eu:` provider: OpenRouter EU in-region routing (`eu.openrouter.ai`, EU/EEA/UK/CH upstreams only, no out-of-region fallback; `OPENROUTER_EU_API_KEY` from a Business-plan org), same gateway handling as `openrouter` (passthrough id, `data_collection: deny`, usage-billed)
- Add `vertex_eu:` provider (`vertex_eu:anthropic/…`, Anthropic models only for now): Claude on Google Vertex AI EU multi-region (`aiplatform.eu.rep.googleapis.com`, `locations/eu`, rawPredict/streamRawPredict), `GOOGLE_CLOUD_PROJECT` + `GOOGLE_APPLICATION_CREDENTIALS` (service-account or ADC JSON, content or path) exchanged for a cached OAuth token; billed as `anthropic` +10%
- Add `anthropic_eu:` provider: Claude on Amazon Bedrock via the `eu.` cross-region inference profile (bedrock-runtime Anthropic Messages path, `AWS_BEARER_TOKEN_BEDROCK`), billed as `anthropic` +10%
- `azure:` provider modernized to the Azure OpenAI (Foundry) v1 API: `https://<AZURE_OPENAI_HOST>.openai.azure.com/openai/v1`, Responses API, no api-version; EU Data Zone deployments named after the OpenAI model id; billed as `openai` +10% via `PRICED_AS`
- Add `openai_eu:` provider: OpenAI EU data residency (`eu.api.openai.com`, `OPENAI_API_KEY`), billed as the `openai` catalog endpoint +10%; alias pricing via `PRICED_AS` table (also covers `vertex`)
- opencode_go: base URL → `https://opencode.ai/inference/go/v1`; Gemini ids use the Gemini-native protocol (`gemini-4-argon` → Go's `gemini-4-argon-contributor`); export flags Go `-contributor` endpoints `trains_on_data`
- Add `vertex:` provider: the Gemini API (`GOOGLE_API_KEY`) under a slug CLIProxyAPI never rebinds, billed with the `google-ai-studio` catalog endpoint — for Google-only metered models such as `gemini-4-argon`
- Fix Novita GLM models 404: map OpenRouter `z-ai/` ids to Novita's `zai-org/` namespace

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
