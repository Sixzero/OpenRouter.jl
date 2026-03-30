# Bug Report: Cached token counts not returned in chat completions response

**Is it a request payload issue?**
[x] Yes, this is a request payload issue. I am using a client/cURL to send a request payload, but I received an unexpected error.

---

**Describe the bug**
When sending requests to `/v1/chat/completions` with Anthropic `cache_control` markers, the response `usage.prompt_tokens_details.cached_tokens` is always `0` — even on repeated identical requests that should hit the cache. Anthropic's native API returns `cache_creation_input_tokens` and `cache_read_input_tokens` in the usage, but these are not being translated into the chat-completions response format.

**CLI Type**
claude code

**Model Name**
`claude-haiku-4-5-20251001`

**LLM Client**
Custom Julia client (OpenRouter.jl)

**Request Information**

Call 1 (cache write — fresh unique system prompt):
```bash
curl -s http://localhost:8317/v1/chat/completions \
  -H "Authorization: Bearer $CLIPROXYAPI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 20,
    "messages": [
      {"role": "system", "content": [{"type": "text", "text": "<~2000 tokens of unique text>", "cache_control": {"type": "ephemeral"}}]},
      {"role": "user", "content": "call1"}
    ]
  }'
```

Response usage:
```json
{"prompt_tokens": 2026, "completion_tokens": 20, "total_tokens": 2046, "prompt_tokens_details": {"cached_tokens": 0}}
```

Call 2 (identical system prompt — should be a cache hit):
```bash
# same payload, only "call2" in user content
```

Response usage:
```json
{"prompt_tokens": 2026, "completion_tokens": 20, "total_tokens": 2046, "prompt_tokens_details": {"cached_tokens": 0}}
```

**Expected behavior**
- Call 1: `prompt_tokens_details.cached_tokens = 0`, but `prompt_tokens` should reflect only non-cached tokens (i.e. be lower than total input), and ideally a `cache_creation_tokens` field should be present
- Call 2: `prompt_tokens_details.cached_tokens` should be `> 0` (e.g. ~2000), and `prompt_tokens` should reflect only the non-cached tail (the new user message, ~10 tokens)

Anthropic's native API returns:
- `cache_creation_input_tokens` on first call
- `cache_read_input_tokens` on subsequent cache hits

These need to be mapped into the chat-completions response, e.g.:
- `cache_read_input_tokens` → `prompt_tokens_details.cached_tokens`
- `prompt_tokens` should be `input_tokens` only (non-cached), not the total

**OS Type**
- OS: Ubuntu 24.04.4 LTS

**Additional context**
The `cache_control` markers are confirmed present in the outgoing request payload (verified by inspection). The issue is purely in the response translation from Anthropic native format → chat-completions format — `cache_creation_input_tokens` and `cache_read_input_tokens` from Anthropic's usage are being dropped/zeroed instead of forwarded.
