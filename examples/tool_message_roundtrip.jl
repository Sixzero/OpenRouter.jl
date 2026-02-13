#=
Tool Message Round-Trip Example
================================
Tests native tool calling end-to-end: send a prompt with tools → get tool_calls back
→ send ToolMessage results → get final response. Runs on each provider schema.
=#

using OpenRouter
using OpenRouter: AbstractMessage, SystemMessage, UserMessage, AIMessage, ToolMessage,
    HttpStreamCallback, Tool
using JSON3

WEATHER_TOOL = [Tool(
    name = "get_weather",
    description = "Get the current weather for a city",
    parameters = Dict(
        "type" => "object",
        "properties" => Dict(
            "city" => Dict("type" => "string", "description" => "City name")
        ),
        "required" => ["city"]
    )
)]

# Fake weather lookup — receives parsed args dict
function fake_weather(args)
    data = Dict("New York" => "72°F, sunny", "San Francisco" => "58°F, foggy", "London" => "55°F, rainy")
    get(data, get(args, "city", "Unknown"), "Unknown city")
end

models = [
    "anthropic:anthropic/claude-sonnet-4.5"       => "Anthropic",
    "google-ai-studio:google/gemini-2.5-flash"    => "Gemini",
    "openai:openai/gpt-5.2"                       => "OpenAI (ResponseSchema)",
    "groq:meta-llama/llama-4-scout"               => "Groq (ChatCompletion)",
]

for (model, label) in models
    println("\n" * "="^60)
    println("  $label  ($model)")
    println("="^60)

    # Turn 1: ask about weather → expect tool_calls
    println("\n--- Turn 1: requesting tool call ---")
    r1 = aigen(
        "What's the weather in New York? Use the get_weather tool. Be brief.",
        model;
        tools=WEATHER_TOOL,
        streamcallback=HttpStreamCallback(; out=stdout, verbose=false)
    )
    println("\nContent: ", repr(r1.content))
    println("Tool calls: ", r1.tool_calls)
    println("Finish reason: ", r1.finish_reason)

    if r1.tool_calls === nothing || isempty(r1.tool_calls)
        println("⚠ No tool calls returned, skipping turn 2")
        continue
    end

    # Turn 2: send tool results back → expect final text response
    println("\n--- Turn 2: sending tool results, expecting final answer ---")
    tc = r1.tool_calls[1]
    tool_msg = ToolMessage(tc, fake_weather)  # parses args, calls fake_weather(; city=...), wraps result
    println("Tool result: ", tool_msg.content)

    conversation = AbstractMessage[
        SystemMessage(content="You are a helpful weather assistant. Be brief."),
        UserMessage(content="What's the weather in New York?"),
        r1,  # AIMessage with tool_calls
        tool_msg,
    ]

    r2 = aigen(
        conversation, model;
        tools=WEATHER_TOOL,
        streamcallback=HttpStreamCallback(; out=stdout, verbose=false)
    )
    println("\nFinal content: ", r2.content)
    println("Tool calls: ", r2.tool_calls)  # should be nothing
end
