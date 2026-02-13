using OpenRouter
using OpenRouter: AbstractMessage, UserMessage, AIMessage, ToolMessage, Tool
using Test
using Base64

# 200x100 PNG with "HELLO" — load from file for clean base64
const IMG_B64 = let
    raw = read(joinpath(@__DIR__, "..", "assets", "hello.png"))
    "data:image/png;base64," * base64encode(raw)
end

SCREENSHOT_TOOL = [Tool(
    name = "take_screenshot",
    description = "Take a screenshot and return the image",
    parameters = Dict(
        "type" => "object",
        "properties" => Dict(
            "region" => Dict("type" => "string", "description" => "Screen region to capture")
        ),
        "required" => ["region"]
    )
)]

# Build a conversation where AI called a tool and we return image result
function make_tool_image_msgs(; with_text::Bool)
    AbstractMessage[
        UserMessage(content="Take a screenshot of the banner and tell me what word is shown"),
        AIMessage(content="", tool_calls=[
            Dict("id" => "call_1", "type" => "function",
                 "function" => Dict("name" => "take_screenshot",
                                    "arguments" => """{"region":"banner"}"""))
        ]),
        ToolMessage(
            content = with_text ? "Screenshot captured successfully" : "",
            tool_call_id = "call_1",
            name = "take_screenshot",
            image_data = [IMG_B64]
        ),
    ]
end

providers = [
    "anthropic:anthropic/claude-haiku-4.5",       # AnthropicSchema
    "openai:openai/gpt-5.2",                      # ResponseSchema
    "google-ai-studio:google/gemini-2.5-flash",   # GeminiSchema
    "groq:meta-llama/llama-4-scout",              # ChatCompletionSchema
]

@testset "Tool Image Results" begin
    for with_text in [true, false]
        label = with_text ? "with text" : "image only"
        @testset "$label" begin
            for provider_model in providers
                @testset "$provider_model" begin
                    try
                        msgs = make_tool_image_msgs(; with_text)
                        resp = aigen(msgs, provider_model;
                            tools = SCREENSHOT_TOOL,
                            sys_msg = "Look at the image from the tool result. Reply with ONLY the single word shown in the image, nothing else.")

                        @test resp isa AIMessage
                        @test !isempty(resp.content)
                        @test resp.tokens !== nothing
                        println("✓ $provider_model ($label): $(resp.content)")
                    catch e
                        @warn "Failed $provider_model ($label)" exception=e
                        @test false
                    end
                end
            end
        end
    end
end
