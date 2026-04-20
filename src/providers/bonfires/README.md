# Bonfires Provider for memorybench

Runs memorybench benchmarks against a local [Bonfires](https://github.com/Ataxia123/bonfires) Delve stack across three retrieval arms.

## Env vars

- `BONFIRES_API_URL` — default `http://localhost:8000`
- `BONFIRES_API_KEY` — local dev key
- `BONFIRES_ARM` — `vector` | `graph` | `smart`
- `BONFIRES_BONFIRE_ID` — default `locomo-eval`

## Judge via OpenRouter

- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_API_KEY=<OpenRouter key>`
- `JUDGE_MODEL=openai/gpt-4o`

## Prerequisites

1. Delve docker stack running (`make up` in `delve/`).
2. Ollama running (`ollama serve`) with `nomic-embed-text` pulled.
3. Neo4j + Weaviate healthy (Delve's `/healthz` covers this).

## Run

```bash
BONFIRES_ARM=smart npm run bench -- --provider bonfires --benchmark locomo
```
