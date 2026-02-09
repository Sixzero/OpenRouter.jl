
"""
    groq_model_transform(model_id::String)::String

Transform OpenRouter model IDs to Groq-specific model IDs.
Handles various model mappings for Groq's native API.

# Examples
```julia
groq_model_transform("moonshotai/kimi-k2-0905")  # => "moonshotai/kimi-k2-instruct-0905"
groq_model_transform("other/model")              # => "other/model"
```
"""
function groq_model_transform(m_id::AbstractString)::AbstractString
    m_id == "moonshotai/kimi-k2-0905" && return "moonshotai/kimi-k2-instruct-0905"
    m_id == "moonshotai/kimi-k2-0905:exacto" && return "moonshotai/kimi-k2-instruct-0905"
    m_id == "meta-llama/llama-4-scout" && return "meta-llama/llama-4-scout-17b-16e-instruct"
    m_id == "meta-llama/llama-4-maverick" && return "meta-llama/llama-4-maverick-17b-128e-instruct"
    m_id == "meta-llama/llama-3.1-8b-instruct" && return "llama-3.1-8b-instant"
    m_id == "meta-llama/llama-3.3-70b-instruct" && return "llama-3.3-70b-versatile"
    m_id == "openai/gpt-oss-120b:exacto" && return "openai/gpt-oss-120b"
    return m_id
end

"""
    openai_model_transform(model_id::String)::String

Transform model IDs for OpenAI. Removes openai/ prefix and handles specific mappings.
"""
function openai_model_transform(model_id::AbstractString)::AbstractString
    # Remove provider prefix first
    base_id = strip_provider_prefix(model_id, "openai")
    
    # Handle specific mappings
    base_id == "gpt-4o:extended" && return "gpt-4o"
    base_id == "codex-mini" && return "codex-mini-latest"
    base_id == "o4-mini-high" && return "o4-mini"
    base_id == "gpt-5-chat" && return "gpt-5-chat-latest"
    base_id == "gpt-5.1-chat" && return "gpt-5.1-chat-latest"
    base_id == "o3-mini-high" && return "o3-mini"
    
    return base_id
end

"""
    anthropic_model_transform(model_id::String)::String

Transform model IDs for Anthropic. Removes anthropic/ prefix and replaces dots with dashes.
Also handles special cases and version matching.
"""
function anthropic_model_transform(model_id::AbstractString)::AbstractString
    # Remove provider prefix first
    base_id = strip_provider_prefix(model_id, "anthropic")
    
    # Remove any :thinking or other suffixes for matching
    base_id = split(base_id, ':')[1]
    
    # Replace dots with dashes in version numbers (e.g., "4.5" -> "4-5")
    base_id = replace(base_id, r"(\d+)\.(\d+)" => s"\1-\2")
    
    # Handle specific model mappings to match native API format
    # Note: We return the base transformed ID, not the full dated version
    # The matching logic should handle partial matching
    base_id == "claude-3-opus" && return "claude-3-opus-20240229"
    base_id == "claude-3-haiku" && return "claude-3-haiku-20240307"
    base_id == "claude-3-5-haiku" && return "claude-3-5-haiku-20241022"
    base_id == "claude-3-7-sonnet" && return "claude-3-7-sonnet-20250219"
    base_id == "claude-sonnet-4" && return "claude-sonnet-4-20250514"
    base_id == "claude-opus-4" && return "claude-opus-4-20250514"
    base_id == "claude-opus-4-1" && return "claude-opus-4-1-20250805"
    base_id == "claude-opus-4-6" && return "claude-opus-4-6"
    base_id == "claude-sonnet-4-5" && return "claude-sonnet-4-5-20250929"
    base_id == "claude-haiku-4-5" && return "claude-haiku-4-5-20251001"
    
    return base_id
end

"""
    google_model_transform(model_id::String)::String

Transform model IDs for Google. Removes google/ prefix.
"""
function google_model_transform(model_id::AbstractString)::AbstractString
    return strip_provider_prefix(model_id, "google")
end

"""
    cerebras_model_transform(model_id::String)::String

Transform model IDs for Cerebras.
"""
function cerebras_model_transform(m_id::AbstractString)::AbstractString
    m_id == "meta-llama/llama-3.1-8b-instruct" && return "llama3.1-8b"
    m_id == "qwen/qwen3-32b" && return "qwen-3-32b"
    m_id == "openai/gpt-oss-120b" && return "gpt-oss-120b"
    m_id == "qwen/qwen3-235b-a22b-2507" && return "qwen-3-235b-a22b-instruct-2507"
    m_id == "meta-llama/llama-3.3-70b-instruct" && return "llama-3.3-70b"
    m_id == "z-ai/glm-4.6" && return "zai-glm-4.6"
    return m_id
end

"""
    together_model_transform(model_id::String)::String

Transform model IDs for Together. Currently returns unchanged.
"""
function together_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    mistral_model_transform(model_id::String)::String

Transform model IDs for Mistral. Currently returns unchanged.
"""
function mistral_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    fireworks_model_transform(model_id::String)::String

Transform model IDs for Fireworks. Currently returns unchanged.
"""
function fireworks_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    deepseek_model_transform(model_id::String)::String

Transform model IDs for DeepSeek. Currently returns unchanged.
"""
const DEEPSEEK_MODEL_MAP = Dict{String,String}(
    # OpenRouter -> native DeepSeek API
    "deepseek/deepseek-v3.2" => "deepseek-chat",
    "deepseek/deepseek-v3.2-speciale" => "deepseek-reasoner",
)

function deepseek_model_transform(model_id::AbstractString)::AbstractString
    key = lowercase(model_id)
    return get(DEEPSEEK_MODEL_MAP, key, model_id)
end

"""
    sambanova_model_transform(model_id::String)::String

Transform model IDs for SambaNova. Currently returns unchanged.
"""
function sambanova_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    xai_model_transform(model_id::String)::String

Transform model IDs for xAI. Removes x-ai/ prefix and handles specific mappings.
"""
function xai_model_transform(m_id::AbstractString)::AbstractString
    # Remove provider prefix first
    base_id = strip_provider_prefix(m_id, "x-ai")

    # Handle specific mappings
    base_id == "grok-4-fast" && return "grok-4-fast-reasoning"
    base_id == "grok-3-mini-beta" && return "grok-3-mini"
    base_id == "grok-4" && return "grok-4-0709"
    base_id == "grok-3-beta" && return "grok-3"

    return base_id
end

"""
    moonshotai_model_transform(model_id::String)::String

Transform model IDs for MoonshotAI. Currently returns unchanged.
"""
function moonshotai_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    minimax_model_transform(model_id::String)::String

Transform model IDs for Minimax. Currently returns unchanged.
"""
function minimax_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    ollama_model_transform(model_id::String)::String

Transform model IDs for Ollama. Currently returns unchanged.
"""
function ollama_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    cohere_model_transform(model_id::String)::String

Transform model IDs for Cohere. Currently returns unchanged.
"""
function cohere_model_transform(model_id::AbstractString)::AbstractString
    return model_id
end

"""
    atlascloud_model_transform(model_id::String)::String

Transform model IDs for AtlasCloud. Maps DeepSeek models to the
provider's `deepseek-ai/...` namespace.
"""
function atlascloud_model_transform(model_id::AbstractString)::AbstractString
    if startswith(model_id, "deepseek/")
        suffix = model_id[(length("deepseek/") + 1):end]
        return "deepseek-ai/" * suffix
    end
    return model_id
end

"""
    siliconflow_model_transform(model_id::String)::String

Map OpenRouter model IDs to SiliconFlow's native IDs (case and prefix
differences). Extend this map as SiliconFlow adds or renames models.
"""
const SILICONFLOW_MODEL_MAP = Dict{String,String}(
    "qwen/qwen3-vl-235b-a22b-instruct" => "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "qwen/qwen3-vl-235b-a22b-thinking" => "Qwen/Qwen3-VL-235B-A22B-Thinking",
    "qwen/qwen3-vl-30b-a3b-thinking" => "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "qwen/qwen3-vl-30b-a3b-instruct" => "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen/qwen3-32b" => "Qwen/Qwen3-32B",
    "qwen/qwen3-30b-a3b" => "Qwen/Qwen3-30B-A3B",
    "qwen/qwen3-30b-a3b-instruct-2507" => "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen/qwen3-30b-a3b-thinking-2507" => "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen/qwen3-235b-a22b-2507" => "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen/qwen3-235b-a22b-thinking-2507" => "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "qwen/qwen3-next-80b-a3b-instruct" => "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "qwen/qwq-32b" => "Qwen/QwQ-32B",
    "qwen/qwen3-coder-30b-a3b-instruct" => "Qwen/Qwen3-Coder-30B-A3B-Instruct",

    "deepseek/deepseek-v3.2" => "deepseek-ai/DeepSeek-V3.2",
    "deepseek/deepseek-v3.2-exp" => "deepseek-ai/DeepSeek-V3.2-Exp",
    "deepseek/deepseek-v3.1-terminus" => "deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepseek/deepseek-chat-v3.1" => "deepseek-ai/DeepSeek-V3.1",

    "moonshotai/kimi-k2-0905" => "moonshotai/Kimi-K2-Instruct-0905",
    "moonshotai/kimi-k2-thinking" => "moonshotai/Kimi-K2-Thinking",
    "moonshotai/kimi-dev-72b" => "moonshotai/Kimi-Dev-72B",

    "tencent/hunyuan-a13b-instruct" => "tencent/Hunyuan-A13B-Instruct",
    "baidu/ernie-4.5-300b-a47b" => "baidu/ERNIE-4.5-300B-A47B",
    "z-ai/glm-4.5-air" => "z-ai/GLM-4.5-Air",
    "z-ai/glm-4.6" => "zai-org/GLM-4.6",
    "meta-llama/llama-3.1-8b-instruct" => "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "minimax/minimax-m1" => "MiniMaxAI/MiniMax-M1-80k",
)

function siliconflow_model_transform(model_id::AbstractString)::AbstractString
    key = lowercase(model_id)
    return get(SILICONFLOW_MODEL_MAP, key, model_id)
end

"""
    strip_provider_prefix(model_id::AbstractString, provider::AbstractString)::AbstractString

Remove provider prefix from model ID if present.
Helper function for model transformations.

# Examples
```julia
strip_provider_prefix("openai/gpt-4", "openai")     # => "gpt-4"
strip_provider_prefix("gpt-4", "openai")            # => "gpt-4"
strip_provider_prefix("google/gemini-pro", "google") # => "gemini-pro"
```
"""
function strip_provider_prefix(model_id::AbstractString, provider::AbstractString)::AbstractString
    provider_prefix = lowercase(provider) * "/"
    return startswith(model_id, provider_prefix) ? model_id[(length(provider_prefix) + 1):end] : model_id
end
