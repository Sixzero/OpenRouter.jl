using OpenRouter

# Simple example: print all available models (ID and name)
models = list_models()

println("Total models: $(length(models))")
for (i, model) in enumerate(models)
  println("[$i] $(model.id)  —  $(model.name)")
end