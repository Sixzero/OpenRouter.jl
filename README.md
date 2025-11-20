# OpenRouter.jl  [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sixzero.github.io/OpenRouter.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sixzero.github.io/OpenRouter.jl/dev/) [![Build Status](https://github.com/sixzero/OpenRouter.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/sixzero/OpenRouter.jl/actions/workflows/CI.yml?query=branch%3Amaster) [![Coverage](https://codecov.io/gh/sixzero/OpenRouter.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sixzero/OpenRouter.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

OpenRouter.jl is an unofficial, modern Julia wrapper around the [OpenRouter](https://openrouter.ai) REST API.  
It focuses on:

- **Streaming-first** inference
- **Always-fresh model metadata** from OpenRouter
- **Provider-aware cost & token accounting (including cache + discounts)**
- A single, simple entrypoint function: `aigen`
- A modular core that is easy to extend from other packages

It does not try to be a full “AI framework” – it is a thin, opinionated layer over the REST API that makes it pleasant to use OpenRouter-hosted models from Julia.  
The library is in active use at **TodoFor.ai**, so it will be maintained and extended over time.

---

## Quick data peek: raw `curl` vs Julia helpers

You can always inspect the raw OpenRouter API with `curl`:

```sh
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/providers | jq '.data[0:3]' 2>/dev/null
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models | jq '.data[0:3]'
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models | jq '.data[0:10] | map({id, name, context_length, top_provider: {context_length: .top_provider.context_length, max_completion_tokens: .top_provider.max_completion_tokens}})'
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/providers | jq '.data | map({name, slug})'
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models/moonshotai/kimi-k2-thinking/endpoints | jq '.'
```

The library provides typed equivalents:

```julia
using OpenRouter

# 1. Endpoints / providers for a specific model
raw_ep_json = list_endpoints_raw("moonshotai/kimi-k2-thinking")
endpoints   = list_endpoints("moonshotai/kimi-k2-thinking")  # -> ModelProviders

# 2. All models
raw_models_json = list_models_raw()
models          = parse_models(raw_models_json)  # or simply:
models2         = list_models()                  # convenience wrapper

# 3. Filtered by provider (e.g. only Groq-hosted models)
groq_models = list_models("groq")

# 4. Known provider slugs
known = list_known_providers()

# 5. All endpoints hosted by a provider
groq_eps = list_provider_endpoints("groq")
```

For day-to-day usage you typically combine these with the cache layer, which **automatically keeps you in sync** with changes on OpenRouter:

```julia
# One-shot sync with OpenRouter (models + endpoints)
update_db(; fetch_endpoints = true)

# Explore locally from the cache
cached_models = list_cached_models()
search_models("llama")
```

---

## Model addressing (no aliases)

OpenRouter.jl does **not** introduce its own model aliases.  
Instead, models are always referred to explicitly as:

```text
provider:author/model_id
```

- `provider` – the upstream provider slug (e.g. `openai`, `groq`, `anthropic`, `google-ai-studio`, `cerebras`, `moonshotai`, …)
- `author` – the model author / namespace (often the same as the provider, e.g. `openai/gpt-5.1`)
- `model_id` – the concrete model identifier on OpenRouter

Examples:

```text
openai:openai/gpt-5.1
groq:moonshotai/kimi-k2-0905
anthropic:anthropic/claude-haiku-4.5
google-ai-studio:google/gemini-2.5-flash
cerebras:openai/gpt-oss-120b
```

This explicit style avoids hidden indirections and keeps logs, configs, and code aligned with OpenRouter’s own model catalog.

---

## Why does this exist when PromptingTools.jl exists?

PromptingTools.jl is a higher-level framework that covers prompting patterns, tooling, and multiple providers.  
OpenRouter.jl has a narrower, complementary goal:

- It is a **cleaned up, modular REST wrapper** for OpenRouter:
  - No internal prompt templating
  - No “one true” workflow
  - Minimal public surface (`aigen`, `aigen_raw`, model / cost utilities)
- It aims for **full modularity**:
  - You can layer your own prompting, RAG, orchestration, or logging on top
  - Third-party packages can extend it (e.g. alternative streaming backends, custom providers, extra metrics) without fighting the core design
- It focuses on **OpenRouter’s model catalog**, pricing, and provider quirks:
  - Automatic model metadata and endpoint discovery
  - Provider-specific model ID mapping
  - Token + cost accounting that understands caching and discounts

In short: PromptingTools is an opinionated “batteries included” toolkit; OpenRouter.jl is a minimal, extensible building block.

---

## Core API: `aigen` and `aigen_raw`

```julia
using OpenRouter

long_text = "Some long user-specific context..."

# Same task across multiple providers / models
response = aigen("$long_text Count to 1-10 in 1 line:", "anthropic:anthropic/claude-haiku-4.5")
response = aigen("$long_text Count to 1-10 in 1 line:", "openai:openai/gpt-5.1")
response = aigen("$long_text Count to 1-10 in 1 line:", "google-ai-studio:google/gemini-2.5-flash")
response = aigen("$long_text Count to 1-10 in 1 line:", "groq:moonshotai/kimi-k2-0905")
response = aigen("$long_text Count to 1-10 in 1 line:", "cerebras:openai/gpt-oss-120b")
```

A more basic example:

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
    "cerebras:openai/gpt-oss-120b";
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
  - **streaming** ⇒ `streamed_request!` + SSE parsing, then reconstructs a **POST-like response** (same shape as a non-streaming call)

This design keeps streaming and non-streaming behavior as similar as possible from the caller’s perspective.

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

`TokenCounts` is opinionated but tries hard to be universal: it chooses a small, consistent set of fields and maps provider-specific names into them.

Extraction is schema-specific:

- OpenAI-compatible (`ChatCompletionSchema`) – reads `usage.prompt_tokens`, `usage.completion_tokens`, and (if present) cached tokens
- Anthropic (`AnthropicSchema`) – maps `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, …
- Gemini (`GeminiSchema`) – maps `promptTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount`, …

Pricing is modeled via:

- `Pricing` – per-token and per-request prices (often in USD / 1 tokens), including:
  - `prompt`, `completion`
  - `input_cache_read`, `input_cache_write`
  - `internal_reasoning`, `input_audio_cache`
  - `discount` (global discount factor when present)
- `ProviderEndpoint` – combines `Pricing` with per-endpoint metadata (context length, parameters, status, etc.)

Cost is computed as:

```julia
cost = calculate_cost(endpoint::ProviderEndpoint, tokens::TokenCounts)
```

and exposed on `AIMessage.cost`.  
Cache-related and reasoning-related prices, as well as discounts, are **first-class** in this calculation – they are not bolted on later.

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

- `update_db()` – refresh OpenRouter models (and optionally endpoints); **automatically keeps you in sync** with newly added or updated models
- `get_model(id; fetch_endpoints = false)` – look up a single cached model, optionally ensuring endpoints are fetched
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
- `streamed_request!` wraps the HTTP streaming loop, reconstructs a **POST-like** response body, and keeps the API compatible with non-streaming calls

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

Current limitations / intentional non-goals:

- **No high-level embeddings API yet** (even though embedding models are parsed and represented)
- **No built-in prompt templating** (keep that in separate, opinionated packages)

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