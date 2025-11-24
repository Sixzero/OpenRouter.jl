using OpenRouter: UserMessage, AIMessage
using OpenRouter

# Example 1: Simple string prompt (gets converted to UserMessage internally)
println("=== Example 1: Simple String Prompt ===")
response1 = aigen("What is the capital of France?", "openai:openai/gpt-5-mini")
println("Response: ", response1)
println()

# Example 2: Using UserMessage explicitly
println("=== Example 2: Explicit UserMessage ===")
user_msg = UserMessage(content="Explain quantum computing in simple terms, in 3 sentences.", name="student")
response2 = aigen(user_msg, "openai:openai/gpt-5-mini")
println("Response: ", response2)
println()

#%%
# Example 3: UserMessage with image data
println("=== Example 3: UserMessage with Image ===")
# Example base64 image data (this is a tiny 1x1 red pixel PNG for demo)
tiny_red_pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

user_msg_with_image = UserMessage(
    content="What do you see in this image?",
    image_data=[tiny_red_pixel],
)

# Note: Use a vision-capable model for image analysis
response3 = aigen(user_msg_with_image, "openai:openai/gpt-5-mini")
println("Response: ", response3)
println()
#%%
# Example 4: Back-and-forth conversation
println("=== Example 4: Multi-turn Conversation ===")
conversation = [
    UserMessage(content="Hi! I'm learning about machine learning. Can you help?",),
    AIMessage(content="Of course! I'd be happy to help you learn about machine learning. What specific aspect would you like to start with?", ),
    UserMessage(content="What's the difference between supervised and unsupervised learning?")
]

# Continue the conversation
response4 = aigen(conversation, "openai:openai/gpt-5-mini"; sys_msg="Answer with 2 or 3 sentences max.")
println("AI Response: ", response4)
println()

# Example 5: Conversation with system message
println("=== Example 5: Conversation with System Message ===")
sys_msg = "You are a helpful programming tutor. Keep responses concise and practical."

conversation_with_context = [
    UserMessage(content="I'm new to Julia programming", name="beginner"),
    AIMessage(content="Great choice! Julia is excellent for scientific computing. What would you like to learn first?", name="julia_tutor"),
    UserMessage(content="How do I create a simple function?", name="beginner")
]

response5 = aigen(conversation_with_context, "openai:openai/gpt-5-mini", sys_msg=sys_msg)
println("AI Response: ", response5)
println()
#%%
sys_msg = "You are a helpful programming tutor. Keep responses concise and practical."
conversation = [
    UserMessage(content="Hi! I'm learning about machine learning. Can you help?",),
    AIMessage(content="Of course! I'd be happy to help you learn about machine learning. What specific aspect would you like to start with?", ),
    UserMessage(content="What's the difference between supervised and unsupervised learning?")
]
response5 = aigen(conversation, "anthropic:anthropic/claude-haiku-4.5", sys_msg=sys_msg, cache=:all)
println("AI Response: ", response5)