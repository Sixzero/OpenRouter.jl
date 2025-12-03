using OpenRouter
using OpenRouter: _aigen_core, get_provider_info, ChatCompletionSchema
using OpenRouter: AbstractMessage, UserMessage, AIMessage, SystemMessage, ToolMessage
using JSON3

# Set your DeepSeek API key
# ENV["DEEPSEEK_API_KEY"] = "your_key_here"

# DeepSeek pricing and endpoint (for direct API calls)
const DEEPSEEK_PRICING = Pricing(
    prompt = "0.00000028",           # $0.28/M input tokens
    completion = "0.0000004",        # $0.4/M output tokens  
    request = "0",
    image = "0",
    web_search = "0",
    internal_reasoning = "0",
    image_output = "0",
    audio = nothing,
    input_audio_cache = nothing,
    input_cache_read = "0.000000028", # $0.028/M cache hit
    input_cache_write = nothing,
    discount = nothing
)

const DEEPSEEK_ENDPOINT = ProviderEndpoint(
    name = "deepseek-chat",
    model_name = "deepseek-chat",
    context_length = 64000,
    pricing = DEEPSEEK_PRICING,
    provider_name = "DeepSeek",
    tag = "deepseek",
    quantization = nothing,
    max_completion_tokens = 8192,
    max_prompt_tokens = nothing,
    supported_parameters = nothing,
    uptime_last_30m = nothing,
    supports_implicit_caching = nothing,
    status = nothing
)

# Tool definitions (same as DeepSeek docs)
const TOOLS = [
    Dict(
        "type" => "function",
        "function" => Dict(
            "name" => "get_date",
            "description" => "Get the current date",
            "parameters" => Dict("type" => "object", "properties" => Dict())
        )
    ),
    Dict(
        "type" => "function",
        "function" => Dict(
            "name" => "get_weather",
            "description" => "Get weather of a location, the user should supply the location and date.",
            "parameters" => Dict(
                "type" => "object",
                "properties" => Dict(
                    "location" => Dict("type" => "string", "description" => "The city name"),
                    "date" => Dict("type" => "string", "description" => "The date in format YYYY-mm-dd")
                ),
                "required" => ["location", "date"]
            )
        )
    )
]

# Mock tool implementations
get_date_mock() = "2025-12-01"
get_weather_mock(; location, date) = "Cloudy 7~13°C"

const TOOL_CALL_MAP = Dict(
    "get_date" => () -> get_date_mock(),
    "get_weather" => (args) -> get_weather_mock(; location=args["location"], date=args["date"])
)

# Run a turn with tool calling loop (thinking mode)
function run_turn(messages::Vector{AbstractMessage}; model="deepseek-chat", verbose=true)
    provider_info = get_provider_info("deepseek")
    sub_turn = 1
    total_cost = 0.0
    while true
        result = _aigen_core(messages, provider_info, model, DEEPSEEK_ENDPOINT;
            tools=TOOLS, thinking=Dict("type" => "enabled"))
        
        ai_msg = AIMessage(ChatCompletionSchema(), result; endpoint=DEEPSEEK_ENDPOINT)
        push!(messages, ai_msg)
        total_cost += something(ai_msg.cost, 0.0)
        
        verbose && println("Turn $sub_turn | tokens=$(ai_msg.tokens) | cost=\$$(round(ai_msg.cost; digits=6))\nreasoning=$(ai_msg.reasoning)\ncontent=$(ai_msg.content)\ntool_calls=$(ai_msg.tool_calls)\n")
        
        isnothing(ai_msg.tool_calls) && break
        
        for tc in ai_msg.tool_calls
            func_name = tc["function"]["name"]
            func_args = JSON3.read(tc["function"]["arguments"], Dict)
            tool_result = isempty(func_args) ? TOOL_CALL_MAP[func_name]() : TOOL_CALL_MAP[func_name](func_args)
            verbose && println("tool result for $func_name: $tool_result\n")
            push!(messages, ToolMessage(content=tool_result, tool_call_id=tc["id"]))
        end
        sub_turn += 1
    end
    verbose && println("Total cost: \$$(round(total_cost; digits=6))")
    return messages
end

# Example: Tool calling with thinking mode
messages = AbstractMessage[UserMessage(content="How's the weather in Hangzhou Tomorrow??")]
run_turn(messages)
#%%
using OpenRouter

task_short = "Give me an optimal implementation of is_prime in 3 lines max don't talk unnecessary."

task_long = "Give me an optimal implementation of is_prime in 50 lines max and then think about it a lot, and rewrite it based on your thoughts"
# resp = aigen(task_short, "gpt5")
the_printer(tokens, cost, elapsed) = begin
    elapsed_str = elapsed !== nothing ? " ($(round(elapsed, digits=2))s)" : ""
    return "\n$tokens Cost: \$$(round(cost, digits=6))$elapsed_str"
end
# Custom hooks example with on_done printing
callback = HttpStreamHooks(
on_done = () -> "\n✓ Generation complete!",  # Returns string to be printed
on_start = () -> "🚀 Starting generation...",
on_meta_usr = the_printer,
on_meta_ai = the_printer,
content_formatter = text -> uppercase(text)  # Make all content uppercase
)
resp = aigen(task_short, "novita:deepseek/deepseek-v3.2", streamcallback=callback)
# resp = aigen("Give me an optimal implementation of is_prime in >100 lines and then think about it a lot, and rewrite it based on your thoughts", "novita:deepseek/deepseek-v3.2")
@show resp.tokens
@show resp.cost
;