using OpenRouter
using Base64

# Generate an image with Google's Gemini 2.5 Flash image model
# Note: thinkingConfig=false disables thinking (image models don't support it)
response = aigen(
    "Generate a cute cartoon cat wearing a space helmet",
    "google-ai-studio:google/gemini-2.5-flash-image";
    thinkingConfig=false
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
            filename = "google_generated_image_$i.png"
            open(filename, "w") do f
                write(f, base64decode(data))
            end
            println("Saved: $filename")
        end
    end
end

println("\nTokens: ", response.tokens)
println("Cost: ", response.cost)
