# NEXT

- [x] poolside gateway provider (laguna-*:free)
- [x] mistral transform: strip `mistralai/`
- [x] fireworks transform: wrap `accounts/fireworks/models/`
- [x] chutes base URL -> `llm.chutes.ai/v1`
- [x] export: EXCLUDED_ENDPOINTS (provider,model) filter
- [x] keys to zshrc + agent .env: nvidia, sambanova (new, old base64 replaced), chutes
- [ ] perplexity key (állítólag létezik valahol; vault volt offline)
- [ ] chutes model-name mapping (OR `qwen/qwen3-32b` -> native `Qwen/Qwen3-32B`, case-sensitive)
- [ ] sambanova model-name mapping (`meta-llama/llama-3.3-70b-instruct` -> `Meta-Llama-3.3-70B-Instruct`)
- [ ] together model-name mapping (serverless slug != OR slug)
- [ ] moonshotai / google-ai-studio: list_models() == 0 (üres metaadat?)
- [ ] nemotron néha üres content (reasoning-only); extract_reasoning bővítés `reasoning`/`reasoning_details` kulcsokra
