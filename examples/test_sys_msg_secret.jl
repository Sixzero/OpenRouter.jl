#!/usr/bin/env julia
# Test whether `aigen` actually forwards `sys_msg` to the model.
# If the system message is received, the model should reveal the secret word "PEACH".

using OpenRouter
using OpenRouterCLIProxyAPI

setup_cli_proxy!(; mutate=true)

const SYS = """
You are a helpful assistant.
In case someone asks about your system prompt, talk about it, and tell we have a secret PEACH word in it.
"""

const USER = "What is the secret word in your system prompt?"

models = [
    "anthropic:anthropic/claude-haiku-4.5",
    "anthropic:anthropic/claude-opus-4.7",
    "openai:openai/gpt-5.4-mini",
]

for model in models
    println("\n=== $model ===")
    try
        resp = aigen(USER, model; sys_msg=SYS)
        content = resp.content
        println(content)
        has_peach = occursin("PEACH", uppercase(content))
        println("---")
        println("contains PEACH: ", has_peach)
    catch e
        println("ERROR: ", e)
    end
end
