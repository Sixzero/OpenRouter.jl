# Tool Use

OpenRouter.jl supports tool calling (function calling) across all four provider schemas: OpenAI, Anthropic, Gemini, and the OpenAI Responses API. Tools are defined once and automatically converted to each provider's native format.

## Defining tools

```julia
using OpenRouter

weather_tool = Tool(
    name = "get_weather",
    description = "Get the current weather for a city",
    parameters = Dict(
        "type" => "object",
        "properties" => Dict(
            "city" => Dict("type" => "string", "description" => "City name")
        ),
        "required" => ["city"]
    )
)
```

Pass tools as a vector:

```julia
tools = [weather_tool]
```

## Calling aigen with tools

```julia
r = aigen("What's the weather in NYC?", "anthropic:anthropic/claude-haiku-4.5";
    tools = tools)

# When the model wants to call a tool:
r.finish_reason  # "tool_use" (Anthropic), "stop" with tool_calls (OpenAI), etc.
r.tool_calls     # Vector{Dict} — normalized across all schemas
```

`tool_calls` is always normalized to the OpenAI format regardless of provider:

```julia
r.tool_calls[1]
# Dict(
#   "id" => "call_abc123",
#   "type" => "function",
#   "function" => Dict("name" => "get_weather", "arguments" => "{\"city\":\"NYC\"}")
# )
```

## Parsing tool call arguments

```julia
using OpenRouter: get_arguments

args = get_arguments(r.tool_calls[1])
# Dict{String,Any}("city" => "NYC")
```

Handles both JSON strings and already-parsed Dicts.

## Creating ToolMessage responses

Three ways to create a `ToolMessage`:

### 1. From keyword arguments

```julia
msg = ToolMessage(
    content = "72°F, sunny",
    tool_call_id = r.tool_calls[1]["id"],
    name = "get_weather"
)
```

### 2. From a tool_call dict + string result

```julia
msg = ToolMessage(r.tool_calls[1], "72°F, sunny")
```

Automatically extracts `tool_call_id` and `name` from the dict.

### 3. From a tool_call dict + function

```julia
function fake_weather(args::Dict{String,Any})
    data = Dict("NYC" => "72°F, sunny", "SF" => "58°F, foggy")
    get(data, args["city"], "Unknown city")
end

msg = ToolMessage(r.tool_calls[1], fake_weather)
```

Parses arguments, calls your function, wraps the return value with `string()`.

## Multi-turn tool conversation

```julia
using OpenRouter: AbstractMessage, SystemMessage, UserMessage, AIMessage, ToolMessage

# Turn 1: model requests tool call
r1 = aigen("What's the weather in NYC?", model; tools=tools)

# Turn 2: send tool result back
tool_msg = ToolMessage(r1.tool_calls[1], fake_weather)

conversation = AbstractMessage[
    UserMessage(content="What's the weather in NYC?"),
    r1,        # AIMessage with tool_calls
    tool_msg,  # ToolMessage with result
]

r2 = aigen(conversation, model; tools=tools)
println(r2.content)  # "It's 72°F and sunny in NYC!"
```

Multiple tool calls in a single response (parallel tool use) are supported — just create a `ToolMessage` for each:

```julia
tool_msgs = [ToolMessage(tc, fake_weather) for tc in r1.tool_calls]
conversation = AbstractMessage[
    UserMessage(content="Weather in NYC and SF?"),
    r1,
    tool_msgs...,
]
```

## Images in tool results

Tools can return images alongside text. Pass base64-encoded images via `image_data`:

```julia
using Base64

img_bytes = read("screenshot.png")
img_b64 = "data:image/png;base64," * base64encode(img_bytes)

msg = ToolMessage(
    content = "Screenshot captured",
    tool_call_id = tc["id"],
    name = "take_screenshot",
    image_data = [img_b64]
)
```

Also works with the convenience constructors:

```julia
msg = ToolMessage(tc, "Screenshot captured"; image_data=[img_b64])
```

Image-only results (no text) are supported too:

```julia
msg = ToolMessage(tc, ""; image_data=[img_b64])
```

### How images are serialized per provider

| Schema | Strategy |
|--------|----------|
| **Anthropic** | Native — image blocks inside `tool_result.content` array |
| **OpenAI / ChatCompletion** | User message with `image_url` parts injected after tool message |
| **Gemini** | `inline_data` parts alongside `functionResponse` |
| **Responses API** | User message with `input_image` entries after `function_call_output` |

All four schemas are tested against live APIs. The `image_data` field accepts `data:image/...;base64,...` URLs or raw base64 strings (defaults to `image/jpeg`).

## Streaming with tools

Tool calls work with streaming — arguments are accumulated across chunks:

```julia
using OpenRouter: HttpStreamCallback

r = aigen("Create a file with a short story.", model;
    tools = tools,
    streamcallback = HttpStreamCallback(; out=stdout))

r.tool_calls  # fully accumulated after stream completes
```

## Cross-provider compatibility

The same `Tool` definition works across all providers:

```julia
for model in [
    "anthropic:anthropic/claude-haiku-4.5",
    "openai:openai/gpt-5.2",
    "google-ai-studio:google/gemini-2.5-flash",
    "groq:meta-llama/llama-4-scout",
]
    r = aigen("What's the weather?", model; tools=tools)
    println("$model: $(r.tool_calls)")
end
```

The `Tool` struct is automatically converted to each provider's native format:
- **OpenAI/ChatCompletion**: `{ type: "function", function: { name, description, parameters } }`
- **Anthropic**: `{ name, description, input_schema }`
- **Gemini**: `{ functionDeclarations: [{ name, description, parameters }] }`
- **Responses API**: `{ type: "function", name, description, parameters }`
