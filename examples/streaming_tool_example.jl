#=
Streaming Tool Calls Example
=============================

Demonstrates streaming with tool calls - the tool_calls arguments are accumulated
from delta chunks (useful for long arguments that stream across multiple chunks).

Uses the schema-agnostic `Tool` struct which auto-converts to the right format.
=#

using OpenRouter
using OpenRouter: HttpStreamCallback, Tool

# Define a file creator tool using the schema-agnostic Tool struct
CREATE_FILE_TOOL = [Tool(
    name = "create_file",
    description = "Create a file with the given content at the specified path",
    parameters = Dict(
        "type" => "object",
        "properties" => Dict(
            "path" => Dict("type" => "string", "description" => "The file path to create"),
            "content" => Dict("type" => "string", "description" => "The content to write to the file")
        ),
        "required" => ["path", "content"]
    )
)]

# Stream with tool calls - ask for long content so arguments stream across chunks
response = aigen(
    "Welcome me and then, use the create_file tool to create /tmp/story.txt with a 20-word short story about a robot learning to paint.",
    "openai:openai/gpt-5.2";                            # ResponseSchema
    # "anthropic:anthropic/claude-sonnet-4.5";           # AnthropicSchema
    # "groq:meta-llama/llama-4-scout";                   # ChatCompletionSchema
    # "google-ai-studio:google/gemini-2.5-flash";        # GeminiSchema
    tools=CREATE_FILE_TOOL,
    streamcallback=HttpStreamCallback(; out=stdout, verbose=false)
)

println("\n--- Response ---")
println("Content: ", response.content)
println("Tool calls: ", response.tool_calls)
println("Finish reason: ", response.finish_reason)
