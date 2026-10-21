
"""
Provider information for OpenRouter API providers.
Contains general information about each provider, not request-specific configuration.
"""

mutable struct ProviderInfo
    base_url::String
    auth_header_format::String        # "Bearer", "Api-Key", "x-api-key", etc.
    api_key_env_var::Union{String,Nothing}  # Full env var name, e.g. "OPENAI_API_KEY"
    default_headers::Dict{String,String}    # Headers always sent with requests
    model_name_transform::Union{Function,Nothing}  # Transform model names for this provider
    schema::AbstractRequestSchema     # Request schema for this provider
    notes::String
    # How a `model(level)` thinking suffix reaches the provider:
    #   :passthrough      - kept on the model id (CLIProxyAPI parses it itself)
    #   :reasoning_effort - stripped, sent as `reasoning_effort = level` (native OpenAI-compatible APIs)
    thinking_suffix::Symbol
end
ProviderInfo(base_url, auth, env, headers, transform, schema, notes) =
    ProviderInfo(base_url, auth, env, headers, transform, schema, notes, :passthrough)

# OpenRouter gateway base URL. Providers pointing here (openrouter, crucible, ...) route
# requests through openrouter.ai and skip native endpoint metadata matching (see parse_provider_model).
const OPENROUTER_GATEWAY_URL = "https://openrouter.ai/api/v1"
# EU in-region routing: decrypted and served only by EU/EEA/UK/CH provider endpoints, no
# fallback out of region. Needs the OpenRouter Business plan on the key's org.
const OPENROUTER_EU_GATEWAY_URL = "https://eu.openrouter.ai/api/v1"
is_openrouter_gateway(info) = info.base_url in (OPENROUTER_GATEWAY_URL, OPENROUTER_EU_GATEWAY_URL)

"""
Route only to OpenRouter upstreams that don't train on inputs. Request-level, so it holds
even if the account's "Enable providers that may train on inputs" setting is flipped back.
"""
function deny_data_collection!(payload::AbstractDict)
    prov = get!(() -> Dict{String,Any}(), payload, "provider")
    prov isa AbstractDict && (prov["data_collection"] = "deny")
    payload
end

# Build an OpenRouter-gateway ProviderInfo. `crucible` etc. are OpenRouter-only upstreams
# exposed as their own provider slug; the full model id (incl. `:free`) is passed through.
openrouter_gateway(notes::String, url::String=OPENROUTER_GATEWAY_URL, env::String="OPENROUTER_API_KEY") = ProviderInfo(
    url, "Bearer", env,
    Dict{String,String}(), nothing, ChatCompletionSchema(), notes)

const PROVIDER_INFO = Dict{String,ProviderInfo}(
    # OpenRouter gateway providers: OpenAI-compatible, model id passed through as author/model[:variant].
    "openrouter" => openrouter_gateway("OpenRouter gateway"),
    "openrouter_eu" => openrouter_gateway("OpenRouter EU in-region routing (Business-plan org key)", OPENROUTER_EU_GATEWAY_URL, "OPENROUTER_EU_API_KEY"),
    "crucible"   => openrouter_gateway("OpenRouter-only upstream (Crucible); routed via openrouter.ai"),
    "stealth"    => openrouter_gateway("OpenRouter cloaked/stealth upstream; routed via openrouter.ai"),
    "poolside"   => openrouter_gateway("Poolside (laguna) upstream; routed via openrouter.ai"),

    # OpenAI-compatible / OpenAI-style providers
    "openai" => ProviderInfo(
        "https://api.openai.com/v1",
        "Bearer",
        "OPENAI_API_KEY",
        Dict{String,String}(),
        openai_model_transform,
        ResponseSchema(),
        "OpenAI Responses API (gpt-5+, o-series)"),
    # OpenAI EU data residency (processed/stored in EEA). Same key: a Global-project
    # key works on the eu host. Needs ZDR/MAM approval on the OpenAI org.
    # Billed as `openai` +10% (OpenAI's data-residency uplift), see PRICED_AS.
    # Not in the catalog export: pricier, reachable only via explicit `openai_eu:`.
    "openai_eu" => ProviderInfo(
        "https://eu.api.openai.com/v1",
        "Bearer",
        "OPENAI_API_KEY",
        Dict{String,String}(),
        openai_model_transform,
        ResponseSchema(),
        "OpenAI Responses API, EU data residency (eu.api.openai.com)"),
    "mistral" => ProviderInfo(
        "https://api.mistral.ai/v1",
        "Bearer",
        "MISTRAL_API_KEY",
        Dict{String,String}(),
        mistral_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "meta" => ProviderInfo(
        "https://api.meta.ai/v1",
        "Bearer",
        "META_API_KEY",
        Dict{String,String}(),
        meta_model_transform,
        ChatCompletionSchema(),
        "Meta Model API (Muse Spark); OpenAI-compatible, also serves Responses API",
        :reasoning_effort),
    "fireworks" => ProviderInfo(
        "https://api.fireworks.ai/inference/v1",
        "Bearer",
        "FIREWORKS_API_KEY",
        Dict{String,String}(),
        fireworks_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "together" => ProviderInfo(
        "https://api.together.xyz/v1",
        "Bearer",
        "TOGETHER_API_KEY",
        Dict{String,String}(),
        together_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "groq" => ProviderInfo(
        "https://api.groq.com/openai/v1",
        "Bearer",
        "GROQ_API_KEY",
        Dict{String,String}(),
        groq_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "deepseek" => ProviderInfo(
        "https://api.deepseek.com/v1",
        "Bearer",
        "DEEPSEEK_API_KEY",
        Dict{String,String}(),
        deepseek_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "cerebras" => ProviderInfo(
        "https://api.cerebras.ai/v1",
        "Bearer",
        "CEREBRAS_API_KEY",
        Dict{String,String}(),
        cerebras_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "sambanova" => ProviderInfo(
        "https://api.sambanova.ai/v1",
        "Bearer",
        "SAMBANOVA_API_KEY",
        Dict{String,String}(),
        sambanova_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "xai" => ProviderInfo(
        "https://api.x.ai/v1",
        "Bearer",
        "XAI_API_KEY",
        Dict{String,String}(),
        xai_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "moonshotai" => ProviderInfo(
        "https://api.moonshot.ai/v1",
        "Bearer",
        "MOONSHOT_API_KEY",
        Dict{String,String}(),
        moonshotai_model_transform,
        ChatCompletionSchema(),
        "Standard OpenAI-compatible API"),
    "opencode_go" => ProviderInfo(
        # `inference/go/v1` serves every Go protocol (chat/completions AND Gemini-native
        # `/models/{id}:generateContent`); the older `zen/go/v1` host only chat.
        "https://opencode.ai/inference/go/v1",
        "Bearer",
        "OPENCODE_API_KEY",
        # Zen Go rejects every request without a session id: 400 MissingSessionID
        # "cannot be routed efficiently". `<session_id>` resolves per request to the
        # todo id (see HEADER_ENV_PLACEHOLDERS), so each conversation gets its own
        # affinity key instead of every user sharing one constant.
        Dict{String,String}("x-opencode-session" => "<session_id>"),
        opencode_go_model_transform,
        ChatCompletionSchema(),
        "OpenCode Go subscription; OpenAI-compatible API with bare model IDs"),
    "minimax" => ProviderInfo(
        "https://api.minimaxi.chat/v1",
        "Bearer",
        "MINIMAX_API_KEY",
        Dict{String,String}(),
        minimax_model_transform,
        ChatCompletionSchema(),
        "Uses custom endpoint paths for different model types"),
    "xiaomi" => ProviderInfo(
        "https://api.xiaomimimo.com/v1",
        "Bearer",
        "MIMO_API_KEY",
        Dict{String,String}(),
        xiaomi_model_transform,
        ChatCompletionSchema(),
        "Xiaomi MiMo native API; OpenAI-compatible, bare model IDs"),

    # Anthropic-style, Cohere, Google AI Studio, etc.
    "anthropic" => ProviderInfo(
        "https://api.anthropic.com",
        "x-api-key",
        "ANTHROPIC_API_KEY",
        Dict("anthropic-version" => "2023-06-01"),
        anthropic_model_transform,
        AnthropicSchema(),
        "Uses Anthropic's native API format"),
    # Claude on Amazon Bedrock, EU: bedrock-runtime's Anthropic Messages API path with
    # an `eu.` cross-region inference profile (routes only within EU regions).
    # Bedrock API key in x-api-key. +10% vs global (see PRICED_AS).
    # (bedrock-mantle lists no Anthropic models for new accounts, so not used.)
    "anthropic_eu" => ProviderInfo(
        "https://bedrock-runtime.eu-central-1.amazonaws.com/anthropic",
        "x-api-key",
        "AWS_BEARER_TOKEN_BEDROCK",
        Dict("anthropic-version" => "2023-06-01"),
        bedrock_eu_model_transform,
        AnthropicSchema(),
        "Claude on Amazon Bedrock, EU inference profile (EU data residency), Messages API"),
    # Claude on Google Vertex AI, EU multi-region (processing stays in the EU).
    # No API keys on Vertex: GOOGLE_APPLICATION_CREDENTIALS = service-account (or gcloud ADC)
    # JSON content or path, exchanged for a cached OAuth token. +10% vs global (see PRICED_AS).
    "vertex_eu" => ProviderInfo(
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/<gcp_project>/locations/eu/publishers/anthropic/models",
        "google-oauth",
        "GOOGLE_APPLICATION_CREDENTIALS",
        Dict{String,String}(),
        vertex_model_transform,
        VertexAnthropicSchema(),
        "Claude on Google Vertex AI, EU multi-region endpoint (EU data residency), rawPredict"),
    "cohere" => ProviderInfo(
        "https://api.cohere.ai/v1",
        "Bearer",
        "COHERE_API_KEY",
        Dict{String,String}(),
        cohere_model_transform,
        ChatCompletionSchema(),  # Assuming Cohere uses OpenAI-compatible format
        "Uses Cohere's native API format"),
    "google-ai-studio" => ProviderInfo(
        "https://generativelanguage.googleapis.com/v1beta",
        "x-goog-api-key",
        "GOOGLE_API_KEY",
        Dict{String,String}(),
        google_model_transform,
        GeminiSchema(),
        "Uses Google's Gemini API format, not OpenAI-compatible"),
    # Same Gemini API and key as `google-ai-studio`, under a slug the CLIProxyAPI
    # override never rebinds: for models only Google serves on metered billing
    # (gemini-4-argon) while `google-ai-studio` is pointed at the proxy. The real
    # Vertex endpoint (aiplatform.googleapis.com) refuses API keys, so this does
    # not reach it; the Gemini API bills the same project and lists the same models.
    "vertex" => ProviderInfo(
        "https://generativelanguage.googleapis.com/v1beta",
        "x-goog-api-key",
        "GOOGLE_API_KEY",
        Dict{String,String}(),
        google_model_transform,
        GeminiSchema(),
        "Google Gemini API (metered), not rebound to CLIProxyAPI"),

    # Other OpenAI-compatible hosts
    "deepinfra" => ProviderInfo(
        "https://api.deepinfra.com/v1/openai",
        "Bearer",
        "DEEPINFRA_API_KEY",
        Dict{String,String}(),
        siliconflow_model_transform,  # DeepInfra uses same cased IDs as SiliconFlow
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    # Nebius Token Factory (NL). Region is per model (FI/FR, but also UK/US), so the
    # provider as a whole is NOT EU-resident; inputs/outputs are stored unless opted out.
    "nebius" => ProviderInfo(
        "https://api.tokenfactory.nebius.com/v1",
        "Bearer",
        "NEBIUS_API_KEY",
        Dict{String,String}(),
        nebius_model_transform,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "hyperbolic" => ProviderInfo(
        "https://api.hyperbolic.xyz/v1",
        "Bearer",
        "HYPERBOLIC_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "novita" => ProviderInfo(
        "https://api.novita.ai/v3/openai",
        "Bearer",
        "NOVITA_API_KEY",
        Dict{String,String}(),
        novita_model_transform,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "siliconflow" => ProviderInfo(
        # SiliconFlow documented base URL uses .com; .cn returns 401 for valid keys
        "https://api.siliconflow.com/v1",
        "Bearer",
        "SILICONFLOW_API_KEY",
        Dict{String,String}(),
        siliconflow_model_transform,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "nvidia" => ProviderInfo(
        "https://integrate.api.nvidia.com/v1",
        "Bearer",
        "NVIDIA_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "perplexity" => ProviderInfo(
        "https://api.perplexity.ai",
        "Bearer",
        "PERPLEXITY_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "z.ai" => ProviderInfo(
        "https://api.z.ai/api/paas/v4",
        "Bearer",
        "ZAI_API_KEY",
        Dict{String,String}(),
        zai_model_transform,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "featherless" => ProviderInfo(
        "https://api.featherless.ai/v1",
        "Bearer",
        "FEATHERLESS_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "clarifai" => ProviderInfo(
        "https://api.clarifai.com/v2",
        "Bearer",
        "CLARIFAI_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Clarifai REST API"),
    "atlascloud" => ProviderInfo(
        "https://api.atlascloud.ai/v1",
        "Bearer",
        "ATLASCLOUD_API_KEY",
        Dict{String,String}(),
        atlascloud_model_transform,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),
    "chutes" => ProviderInfo(
        "https://llm.chutes.ai/v1",
        "Bearer",
        "CHUTES_API_KEY",
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "OpenAI-compatible API"),

    # Cloudflare Workers AI; base URL is account-specific (requires CLOUDFLARE_ACCOUNT_ID)
    "cloudflare" => ProviderInfo(
        "https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/v1",
        "Bearer",
        "CLOUDFLARE_API_TOKEN",
        Dict{String,String}(),
        cloudflare_model_transform,
        ChatCompletionSchema(),
        "Requires CLOUDFLARE_ACCOUNT_ID in base URL; models use @cf/ prefix"),

    # Azure OpenAI (Microsoft Foundry) v1 API: no api-version, OpenAI-style paths.
    # Host = our own resource (AZURE_OPENAI_HOST = resource name). Region is the
    # resource + deployment type; we only create Data Zone (EU) deployments, so this
    # slug is EU-resident. `model` = deployment name: name deployments after the
    # OpenAI model id (e.g. `gpt-6.1-sol`). Billed as `openai` +10% (Data Zone
    # uplift vs Global), see PRICED_AS. Excluded from the catalog export.
    "azure" => ProviderInfo(
        "https://<resource>.openai.azure.com/openai/v1",
        "api-key",
        "AZURE_OPENAI_API_KEY",
        Dict{String,String}(),
        openai_model_transform,
        ResponseSchema(),
        "Azure OpenAI v1 Responses API, EU Data Zone deployments (AZURE_OPENAI_HOST = resource name)"),

    # Local / native providers
    "ollama" => ProviderInfo(
        "http://localhost:11434/api",
        "Bearer",                     # Ignored; no auth_header needed when api_key is empty
        nothing,                      # No env var; api_key optional
        Dict{String,String}(),
        ollama_model_transform,
        OllamaSchema(),
        "Local Ollama instance using the native /api/chat API"),
    "ollama_cloud" => ProviderInfo(
        "https://ollama.com/api",
        "Bearer",
        "OLLAMA_API_KEY",
        Dict{String,String}(),
        ollama_model_transform,
        OllamaSchema(),
        "Ollama Cloud (ollama.com) using the native /api/chat API"),
    # ─── Echo providers for E2E testing ───
    "echo_chat" => ProviderInfo(
        "http://localhost:8787/v1",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ChatCompletionSchema(),
        "Echo server for testing ChatCompletionSchema"),
    "echo_anthropic" => ProviderInfo(
        "http://localhost:8787",
        "x-api-key",
        nothing,
        Dict("anthropic-version" => "2023-06-01"),
        nothing,
        AnthropicSchema(),
        "Echo server for testing AnthropicSchema"),
    "echo_gemini" => ProviderInfo(
        "http://localhost:8787",
        "x-goog-api-key",
        nothing,
        Dict{String,String}(),
        nothing,
        GeminiSchema(),
        "Echo server for testing GeminiSchema"),
    "echo_responses" => ProviderInfo(
        "http://localhost:8787",
        "Bearer",
        nothing,
        Dict{String,String}(),
        nothing,
        ResponseSchema(),
        "Echo server for testing ResponseSchema"),
)

"""
    set_provider!(name::String, base_url::String, auth_header_format::String="Bearer",
                 api_key_env_var::Union{String,Nothing}=nothing,
                 default_headers::Dict{String,String}=Dict{String,String}(),
                 model_name_transform::Union{Function,Nothing}=nothing,
                 schema::AbstractRequestSchema=ChatCompletionSchema(),
                 notes::String="Custom provider")

Set/override a provider in the registry. Use this when intentionally overwriting an existing provider.
"""
function set_provider!(name::String, base_url::String,
                      auth_header_format::String="Bearer",
                      api_key_env_var::Union{String,Nothing}=nothing,
                      default_headers::Dict{String,String}=Dict{String,String}(),
                      model_name_transform::Union{Function,Nothing}=nothing,
                      schema::AbstractRequestSchema=ChatCompletionSchema(),
                      notes::String="Custom provider")
    PROVIDER_INFO[lowercase(name)] = ProviderInfo(
        base_url, auth_header_format, api_key_env_var, default_headers, model_name_transform, schema, notes
    )
end

"""
    add_provider(name::String, base_url::String, auth_header_format::String="Bearer",
                api_key_env_var::Union{String,Nothing}=nothing,
                default_headers::Dict{String,String}=Dict{String,String}(),
                model_name_transform::Union{Function,Nothing}=nothing,
                schema::AbstractRequestSchema=ChatCompletionSchema(),
                notes::String="Custom provider")

Add a provider to the registry. Warns if overwriting an existing provider.

# Example
```julia
add_provider("echo", "http://localhost:8080/v1", "Bearer", "ECHO_API_KEY")
add_provider("ollama", "http://localhost:11434/v1", "Bearer")
```
"""
function add_provider(name::String, base_url::String,
                     auth_header_format::String="Bearer",
                     api_key_env_var::Union{String,Nothing}=nothing,
                     default_headers::Dict{String,String}=Dict{String,String}(),
                     model_name_transform::Union{Function,Nothing}=nothing,
                     schema::AbstractRequestSchema=ChatCompletionSchema(),
                     notes::String="Custom provider")
    key = lowercase(name)
    if haskey(PROVIDER_INFO, key)
        @warn "Overwriting existing provider '$name'. Use set_provider! to suppress this warning."
    end
    set_provider!(name, base_url, auth_header_format, api_key_env_var, default_headers, model_name_transform, schema, notes)
end

"""
Transform model name according to provider-specific rules.
"""
function transform_model_name(provider_info::ProviderInfo, model_id::AbstractString)
    return provider_info.model_name_transform === nothing ? model_id : provider_info.model_name_transform(model_id)
end

"""
    remove_provider(name::String)

Remove a provider from the registry.
"""
function remove_provider(name::String)
    delete!(PROVIDER_INFO, lowercase(name))
end

"""
    list_providers()

List all registered providers.
"""
function list_providers()
    return collect(keys(PROVIDER_INFO))
end

"""
Get the provider info for a given slug, or `nothing` if unknown.
"""
function get_provider_info(provider_slug_or_alias::AbstractString)::Union{ProviderInfo,Nothing}
    provider_slug = resolve_model_alias(provider_slug_or_alias)
    slug = lowercase(provider_slug)
    info = get(PROVIDER_INFO, slug, nothing)
    # Fallback: try the separator-free form (e.g. "moonshot-ai" -> "moonshotai",
    # "z-ai" -> "z.ai") to handle OpenRouter's inconsistent provider_name
    # casing/punctuation across models.
    if info === nothing
        target = replace(slug, r"[-._]" => "")
        for (k, v) in PROVIDER_INFO
            replace(lowercase(k), r"[-._]" => "") == target && (info = v; break)
        end
    end
    return info
end

"Get just the base URL for a provider slug, or `nothing` if unknown."
function get_provider_base_url(provider_slug::AbstractString)::Union{String,Nothing}
    info = get_provider_info(provider_slug)
    return info === nothing ? nothing : info.base_url
end

const BASE_URL_ENV_PLACEHOLDERS = Dict(
    "<account_id>" => "CLOUDFLARE_ACCOUNT_ID",
    "<resource>"   => "AZURE_OPENAI_HOST",
    "<gcp_project>" => "GOOGLE_CLOUD_PROJECT",
)

"""Resolve env-var placeholders (e.g. `<account_id>`) in a base URL. No-op for URLs without `<`."""
function resolve_base_url(base_url::AbstractString)::String
    '<' in base_url || return string(base_url)
    url = string(base_url)
    for (placeholder, env_var) in BASE_URL_ENV_PLACEHOLDERS
        contains(url, placeholder) || continue
        val = get(ENV, env_var, "")
        isempty(val) && throw(ArgumentError("Environment variable $env_var is required (for placeholder $placeholder in base URL)"))
        url = replace(url, placeholder => val)
    end
    return url
end

"Build an auth header pair (name => value) for a provider + API key, or `nothing` if provider is unknown."
function get_provider_auth_header(provider_slug::AbstractString,
        api_key::AbstractString)::Union{Pair{String,String},Nothing}
    info = get_provider_info(provider_slug)
    info === nothing && return nothing
    return get_provider_auth_header(info, api_key)
end

"Build an auth header pair (name => value) for a ProviderInfo + API key."
function get_provider_auth_header(info::ProviderInfo, api_key::AbstractString)::Pair{String,String}
    return if info.auth_header_format == "Bearer"
        "Authorization" => "Bearer $api_key"
    elseif info.auth_header_format == "x-api-key"
        "x-api-key" => api_key
    elseif info.auth_header_format == "api-key"
        "api-key" => api_key
    elseif info.auth_header_format == "x-goog-api-key"
        "x-goog-api-key" => api_key
    elseif info.auth_header_format == "google-oauth"  # api_key = credentials JSON or path
        "Authorization" => "Bearer $(google_access_token(api_key))"
    else
        "Authorization" => "$(info.auth_header_format) $api_key"
    end
end

# Placeholders substituted into `default_headers` values at request time.
# `<session_id>` is the caller's conversation id, so a session-affine provider
# (CLIProxyAPI routing.session-affinity, OpenCode Zen) pins all turns of one
# conversation to one upstream credential and its prompt cache. Resolution order:
#   1. `task_local_storage(:llm_session_id)` — per-task, so one process serving many
#      conversations concurrently (the TODOforAI agent) pins each to its own todo id;
#   2. ENV TODOFORAI_TODO_ID — single-conversation processes (edge tools, scripts);
#   3. the fallback: `<session_id>` → a constant (OpenCode Zen rejects an empty one);
#      `<session_id?>` → empty, and `build_headers` DROPS empty headers, so a
#      session-affine proxy sees no session at all (and uses its own heuristics)
#      instead of every unrelated call sharing one bogus binding.
const SESSION_ID_KEY = :llm_session_id
const HEADER_ENV_PLACEHOLDERS = Dict(
    "<session_id>"  => ("TODOFORAI_TODO_ID", "openrouterjl"),
    "<session_id?>" => ("TODOFORAI_TODO_ID", ""),
)

"Pin this task's LLM requests to one conversation (see `<session_id>`)."
set_session_id!(id::AbstractString) = task_local_storage(SESSION_ID_KEY, String(id))
current_session_id()::String = String(get(task_local_storage(), SESSION_ID_KEY, ""))

"""Resolve `<placeholder>`s in a header value. No-op without `<`."""
function resolve_header_value(value::AbstractString)::String
    '<' in value || return string(value)
    out = string(value)
    for (placeholder, (env_var, fallback)) in HEADER_ENV_PLACEHOLDERS
        contains(out, placeholder) || continue
        val = current_session_id()
        isempty(val) && (val = get(ENV, env_var, ""))
        out = replace(out, placeholder => isempty(val) ? fallback : val)
    end
    return out
end

"""
Build complete headers for a provider request.
"""
function build_headers(provider_info::ProviderInfo, api_key::AbstractString)
    headers = ["Content-Type" => "application/json"]
    
    # Add auth header only if provider expects one and api_key is non-empty
    if !isempty(api_key) && provider_info.auth_header_format != ""
        auth_header = get_provider_auth_header(provider_info, api_key)
        push!(headers, auth_header)
    end
    
    # Add default headers for this provider; a placeholder resolving to "" drops the header
    for (k, v) in provider_info.default_headers
        val = resolve_header_value(v)
        isempty(val) || push!(headers, k => val)
    end
    
    return headers
end

"Return the configured API key env var name for a provider, or `nothing`."
function get_provider_env_var_name(provider_slug::AbstractString)::Union{String,Nothing}
    info = get_provider_info(provider_slug)
    return info === nothing ? nothing : info.api_key_env_var
end

"List all known provider slugs."
function list_known_providers()::Vector{String}
    return collect(keys(PROVIDER_INFO))
end

"Check if this provider slug is known."
function is_known_provider(provider_slug::AbstractString)::Bool
    return haskey(PROVIDER_INFO, provider_slug)
end

"""
    extract_provider_from_model(model_name::String) -> String

Extract provider name from model name in format "provider:author/model_id" or fallback to "openai".

# Examples
```julia
extract_provider_from_model("openai:openai/gpt-4") # => "openai"
extract_provider_from_model("anthropic:anthropic/claude-3-5-sonnet") # => "anthropic"
extract_provider_from_model("cerebras:meta-llama/llama-3.1-8b") # => "cerebras"
extract_provider_from_model("gpt-4") # => "openai" (fallback)
```
"""
function extract_provider_from_model(model_name::String)
    model_name = resolve_model_alias(model_name)
    colon_idx = findfirst(':', model_name)
    if colon_idx !== nothing
        return lowercase(model_name[1:colon_idx-1])
    end
    @warn "Provider was missing from slug: $model_name"
    return "openai"
end



"""
Get the appropriate schema for a provider info and model.
"""
function get_provider_schema(provider_info::ProviderInfo, model_id::AbstractString)::AbstractRequestSchema
    # OpenCode Go exposes Gemini models only on the Gemini-native protocol
    # (chat/completions answers ModelProtocolUnsupported); the rest stay OpenAI-style.
    provider_info === get(PROVIDER_INFO, "opencode_go", nothing) && startswith(model_id, "gemini-") &&
        return GeminiSchema()
    return provider_info.schema
end

"""
    create_stub_endpoint(provider_name, model_id; pricing=nothing)

Create a stub ProviderEndpoint for local/non-OpenRouter providers.
"""
function create_stub_endpoint(provider_name::AbstractString, model_id::AbstractString; pricing=nothing)
    ProviderEndpoint(; name=model_id, model_name=model_id, pricing, provider_name=string(provider_name),
        context_length=nothing, tag=nothing, quantization=nothing, max_completion_tokens=nothing,
        max_prompt_tokens=nothing, supported_parameters=nothing, uptime_last_30m=nothing,
        supports_implicit_caching=nothing, status=nothing)
end

const ZERO_PRICING = Pricing("0", "0", "0", "0", "0", "0", nothing, nothing, nothing, "0", nothing, nothing)

"""Create stub with zero pricing (for Ollama etc. where we want cost tracking)."""
create_stub_endpoint_zero_pricing(provider_name, model_id) = 
    create_stub_endpoint(provider_name, model_id; pricing=ZERO_PRICING)
