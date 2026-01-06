using OpenRouter
using Base64

# Generate an image with OpenAI's gpt-5 model using image_generation tool
# The Responses API uses mainline models (gpt-5) with tools, not dedicated image models
response = aigen(
    "Generate a cute cartoon cat wearing a space helmet",
    "openai:openai/gpt-5";
    tools=[Dict("type" => "image_generation")]
)

println("Response content: ", response.content)
println("Generated images: ", length(something(response.image_data, String[])))

# Save the generated image if present
if response.image_data !== nothing && !isempty(response.image_data)
    for (i, img) in enumerate(response.image_data)
        # Extract base64 data from data URL
        if startswith(img, "data:")
            # Format: "data:image/png;base64,<data>"
            data = split(img, ",", limit=2)[2]
            filename = "generated_image_$i.png"
            open(filename, "w") do f
                write(f, base64decode(data))
            end
            println("Saved: $filename")
        end
    end
end

#%%
# Image generation with quality and transparency options
response2 = aigen(
    "A serene Japanese garden with cherry blossoms and a koi pond",
    "openai:openai/gpt-5";
    tools=[Dict("type" => "image_generation", "quality" => "low")]
)
@show response2.content

println("\nSecond image generation:")
println("Tokens: ", response2.tokens)
println("Cost: ", response2.cost)
