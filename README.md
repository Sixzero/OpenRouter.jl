# OpenRouter [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sixzero.github.io/OpenRouter.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sixzero.github.io/OpenRouter.jl/dev/) [![Build Status](https://github.com/sixzero/OpenRouter.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/sixzero/OpenRouter.jl/actions/workflows/CI.yml?query=branch%3Amaster) [![Coverage](https://codecov.io/gh/sixzero/OpenRouter.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sixzero/OpenRouter.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)



# Sneak peek into the data

```
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/providers | jq '.data[0:3]' 2>/dev/null
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models | jq '.data[0:3]
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models | jq '.data[0:10] | map({id, name, context_length, top_provider: {context_length: .top_provider.context_length, max_completion_tokens: .top_provider.max_completion_tokens}})'
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/providers | jq '.data | map({name, slug})'
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models/moonshotai/kimi-k2-thinking/endpoints | jq '.'
```