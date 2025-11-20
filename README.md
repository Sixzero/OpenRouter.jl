# OpenRouter.jl  [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sixzero.github.io/OpenRouter.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sixzero.github.io/OpenRouter.jl/dev/) [![Build Status](https://github.com/sixzero/OpenRouter.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/sixzero/OpenRouter.jl/actions/workflows/CI.yml?query=branch%3Amaster) [![Coverage](https://codecov.io/gh/sixzero/OpenRouter.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sixzero/OpenRouter.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

OpenRouter.jl is an unofficial, modern Julia client for the [OpenRouter](https://openrouter.ai) REST API, focused on:

- **Streaming-first** inference
- **Always-fresh model metadata** from OpenRouter
- **Provider-aware cost & token accounting**
- A single, simple entrypoint function: `aigen`

It is inspired by `ai-sdk`-style LLM clients, but tailored to Julia and OpenRouter’s ecosystem.  
The library is in active use at **TodoFor.ai**, so it will be maintained and extended over time.

---

## Model addressing (no aliases)

OpenRouter.jl does **not** introduce its own model aliases.  
Instead, models are always referred to explicitly as:

```text
provider:author/model_id
```

- `provider` – the upstream provider slug (e.g. `openai`, `groq`, `anthropic`, `google-ai-studio`, `cerebras`, `moonshotai`, …)
- `author` – the model author / namespace (often the same as provider, e.g. `openai/gpt-5.1`)
- `model_id` – the concrete model identifier on OpenRouter

Examples:

```text
openai:openai/gpt-5.1
groq:qwen/qwen3-222b
anthropic:anthropic/claude-3.5-haiku
google-ai-studio:google/gemini-2.0-flash
cerebras:meta-llama/llama-3.1-8b-instruct
```

This explicit style avoids hidden indirections and keeps logs, configs, and code aligned with OpenRouter’s own model catalog.

---

## Why a dedicated Julia client?

Compared to more general tools (like PromptingTools.jl), OpenRouter.jl is intentionally focused:

- **Always up-to-date model list**
  - Uses OpenRouter’s model APIs + a local scratch cache
  - `update_db()` and `get_model()` keep you in sync with newly added models and endpoints

- **No prompt templating**
  - Prompt templating is opinionated; this package stays neutral
  - You are free to use your own prompt libraries on top (or write your own)

- **Streaming-first design**
  - Streaming is a first-class path, not an afterthought
  - Works for multiple providers, including **Cerebras** and **Google Gemini**
  - Reasoning / thinking token support is being extended as providers evolve

- **No embeddings (yet)**
  - Embedding models and their metadata are parsed and understood, but high-level embedding APIs are not exposed yet

- **First-class token and pricing support**
  - `TokenCounts` introduced as a universal struct for token accounting across providers
  - Pricing, caching, and discount pricing are modeled and computed via `Pricing`, `ProviderEndpoint`, and `calculate_cost`

- **Simple public surface API**
  - `aigen_raw` – low-level, returns raw JSON plus metadata
  - `aigen` – high-level, returns an `AIMessage` with `content`, `tokens`, `elapsed`, and `cost`

---

## Core API: `aigen` and `aigen_raw`

```julia
using OpenRouter

# Basic call
msg = aigen(
    "Write a short introduction to Julia for Python users.",
    "openai:openai/gpt-5.1";
    sys_msg = "You are a concise technical writer."
)

println(msg.content)
println("Tokens: ", msg.tokens)
println("Cost (USD): ", msg.cost)
println("Elapsed (s): ", msg.elapsed)
```

### Streaming

Streaming swaps a single HTTP request for a streaming HTTP request; the rest of the pipeline is shared:

```julia
using OpenRouter

callback = HttpStreamCallback(; out = stdout)

msg = aigen(
    "Count to 20, one number per line.",
    "cerebras:meta-llama/llama-3.1-8b-instruct";
    stream_callback = callback,
)

println()
println("Final message: ", msg.content)
println("Tokens: ", msg.tokens)
println("Cost: ", msg.cost)
```

Under the hood:

- `aigen` calls `aigen_raw`
- `_aigen_core` branches only at the HTTP layer:
  - **non-streaming** ⇒ `HTTP.post`
  - **streaming** ⇒ `streamed_request!` + SSE parsing, then reconstructs a normal-like response

This design keeps streaming and non-streaming behavior as similar as possible from the caller’s perspective.

---

## Sneak peek into OpenRouter data (via this package)

The README “curl” examples show what exists on the wire. With OpenRouter.jl you can do the same from Julia and get rich types.

### List models

```julia
using OpenRouter

# Refresh local cache from OpenRouter
update_db()

# List cached models (OpenRouterModel structs)
models = list_cached_models()
first(models, 3)
```

Equivalent to:

```sh
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models | \
  jq '.data[0:3]'
```

### Inspect endpoints & pricing

```julia
using OpenRouter

# Ensure endpoints are fetched and cached
update_db(; fetch_endpoints = true)

cached = get_model("openai/gpt-5.1"; fetch_endpoints = true)
eps = cached.endpoints.endpoints

# Inspect first endpoint
ep = eps[1]
ep.provider_name
ep.context_length
ep.pricing
```

Similar to:

```sh
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models | \
  jq '.data[0:10] | map({id, name, context_length, top_provider: {context_length: .top_provider.context_length, max_completion_tokens: .top_provider.max_completion_tokens}})'
```

### List providers

```julia
using OpenRouter

list_known_providers()
```

Which aligns with:

```sh
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/providers | \
  jq '.data | map({name, slug})'
```

### Inspect provider-specific endpoints

```julia
using OpenRouter

cached = get_model("moonshotai/kimi-k2-thinking"; fetch_endpoints = true)
cached.endpoints.endpoints
```

Roughly corresponding to:

```sh
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models/moonshotai/kimi-k2-thinking/endpoints | jq '.'
```

---

## Token accounting & cost calculation

OpenRouter.jl normalizes provider-specific usage fields into a single struct:

```julia
TokenCounts(
    prompt_tokens::Int,
    completion_tokens::Int,
    total_tokens::Int,
    input_cache_read::Int,
    input_cache_write::Int,
    internal_reasoning::Int,
    input_audio_cache::Int,
)
```

Extraction is schema-specific:

- OpenAI-compatible (`ChatCompletionSchema`) – reads `usage.prompt_tokens`, `usage.completion_tokens`, etc.
- Anthropic (`AnthropicSchema`) – maps `input_tokens`, `output_tokens`, cache fields, etc.
- Gemini (`GeminiSchema`) – maps `promptTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount`, etc.

Pricing is modeled via:

- `Pricing` – per-token and per-request prices (often in USD per 1K tokens)
- `ProviderEndpoint` – combines `Pricing` with per-endpoint metadata (context length, parameters, status, etc.)

Cost is then computed as:

```julia
cost = calculate_cost(endpoint::ProviderEndpoint, tokens::TokenCounts)
```

and exposed on `AIMessage.cost`.

Cache-related and reasoning-related prices (`input_cache_read`, `input_cache_write`, `internal_reasoning`, `input_audio_cache`, and `discount`) are first-class citizens in these calculations.

---

## Provider-specific model mapping

A subtle but important OpenRouter quirk:

- OpenRouter exposes **its own model IDs** (e.g. `openai/gpt-5.1`)
- Individual providers (OpenAI, Anthropic, Groq, Cerebras, xAI, …) sometimes expose **different IDs** or slightly different naming/versioning for the same underlying model

Because of this, OpenRouter.jl includes a provider mapping layer in `src/model_mapping.jl`:

- `openai_model_transform`
- `anthropic_model_transform`
- `groq_model_transform`
- `cerebras_model_transform`
- `xai_model_transform`
- others…

These functions:

- Strip provider prefixes where needed (e.g. `openai/gpt-4o` → `gpt-4o`)
- Normalize versioning where required (e.g. Anthropic’s dated model IDs)
- Map special or deprecated IDs into currently supported upstream IDs
- Are intended to be **automatically updated by tooling / AI assistance** as new models and edge cases appear

This allows you to use **one consistent OpenRouter-flavored model ID** and still transparently call the native provider APIs.

---

## Storage & custom models

OpenRouter.jl keeps a lightweight local cache (via `Scratch.jl`):

- `update_db()` – refresh OpenRouter models (and optionally endpoints)
- `get_model(id; fetch_endpoints = false)` – lookup a single cached model
- `list_cached_models()` – list all cached `OpenRouterModel`s
- `search_models(query)` – simple substring search in cached model IDs & names

You can also define **custom models** for local experimentation:

```julia
using OpenRouter

add_custom_model(
    "echo/100tps",
    "Echo 100 TPS",
    "Fast echo model for testing",
    8192,
)

add_model(  # alias form, same idea
    "ollama/llama3",
    "Local Llama 3",
    "Self-hosted Llama 3",
    4096,
)
```

These are purely local to your environment and can be used for testing alongside real OpenRouter-backed models.

---

## Streaming implementation & extensibility

Streaming is currently implemented using `HTTP.jl` and Server-Sent Events (SSE):

- `HttpStreamCallback` handles incremental chunks, printing or processing as they arrive
- `streamed_request!` wraps the HTTP streaming loop, reconstructs a final response body, and keeps the API compatible with non-streaming calls

The design goal is to **keep streaming pluggable**:

- The streaming abstraction (`AbstractLLMStream` + callbacks + schema-specific chunk parsing) is part of the package today
- It is intentionally written so that alternative transports (e.g. `LibCurl.jl` or other HTTP clients) can be dropped in with minimal changes
- Over time, this layer may be factorized into a small, independent streaming library if that proves useful

---

## Status and roadmap

The library aims to handle modern OpenRouter + provider features as they appear, with a bias for:

- Correct streaming behavior across providers
- Accurate pricing / token accounting, including caching & discounts
- Robust handling of new model naming schemes via `model_mapping.jl`

Planned / TODO items include:

- **Ollama integration** for local models
- **Echo models** for robust testing of client behavior
- **Embedding APIs** (on top of the already present `OpenRouterEmbeddingModel` types)
- **Extended reasoning support** as more providers expose explicit reasoning / thinking token accounting

---

## License and support

- Licensed under the same terms as this repository (see `LICENSE`)
- Unofficial, but actively used and maintained by **TodoFor.ai**
- Issues and PRs are welcome on GitHub