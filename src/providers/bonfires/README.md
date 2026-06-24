# Bonfires Provider For MemoryBench

Runs MemoryBench benchmarks against local Bonfires services.

The current MemoryKernel/conv-26 quality path is the `hypermem` arm with
`BONFIRES_SEARCH_ENDPOINT=memory-kernel`. The arm name is historical: with
direct indexing enabled, it posts LoCoMo stack-shaped batches directly to
`/search/memory-kernel/index` and searches `/search/memory-kernel`. It does not
use Delve stack V2, `/agents/{id}/stack/process`, or an arq worker.

## Env vars

- `BONFIRES_API_URL` — default `http://localhost:8000`
- `BONFIRES_API_KEY` — local dev key
- `BONFIRES_ARM` — use `hypermem` for MemoryKernel runs
- `BONFIRES_SEARCH_ENDPOINT` — use `memory-kernel`
- `BONFIRES_HYPERMEM_DIRECT_INDEX` — keep `1` unless testing legacy stack ingestion
- `BONFIRES_HYPERMEM_PROFILE` — default `nlp_single_graph_v1`
- `BONFIRES_BONFIRE_ID` — use a descriptive run id

## Judge via OpenRouter

- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_API_KEY=<OpenRouter key>`
- `JUDGE_MODEL=gpt-4.1-mini`

## Prerequisites

1. Local Bonfires dependencies are running: Mongo, Redis, Neo4j, and the configured embedder/reranker.
2. A local Delve server is running with `memory_kernel/src` first on `PYTHONPATH`.
3. `/healthz` succeeds on `BONFIRES_API_URL`.

For current conv-26 MemoryKernel work, the server env should include:

```bash
HYPERMEM_NLP_QUEUE_SUMMARY_MODE=procedural
HYPERMEM_NLP_NORMALIZATION_MODE=per_message
HYPERMEM_GRAMMAR_STATEMENT_CENTROIDS=0
PYTHONPATH=/home/at0x/Vaults/Bonfires/memory_kernel/src:/home/at0x/Vaults/Bonfires/delve/src
```

Leave `HYPERMEM_BOUNDARY_MODE` unset unless you are deliberately testing
episode segmentation; the default for `nlp_single_graph_v1` is
`centroid_timestamp`.

## Run

```bash
cd /home/at0x/Vaults/Bonfires/memorybench
OPENAI_API_KEY="$OPENROUTER_API_KEY" \
OPENAI_BASE_URL="https://openrouter.ai/api/v1" \
BONFIRES_BONFIRE_ID="<descriptive-conv-26-run-id>" \
BONFIRES_API_URL="http://localhost:8000" \
BONFIRES_API_KEY="local-dev-api-key" \
BONFIRES_ARM="hypermem" \
BONFIRES_SEARCH_ENDPOINT="memory-kernel" \
BONFIRES_HYPERMEM_DIRECT_INDEX=1 \
BONFIRES_HYPERMEM_PROFILE="nlp_single_graph_v1" \
LOCOMO_CONV="conv-26" \
bun run src/index.ts run \
  --provider bonfires --benchmark locomo \
  --run-id "<descriptive-id>" \
  --limit 9999 \
  --concurrency 5 \
  --force
```

Correct MemoryKernel indexing prints:

```text
Sampling selected 199 questions from 1986 total
hypermem direct stack index: draining <N> messages across <M> sessions
```

`LOCOMO_CONV` is only applied through MemoryBench's sampling/limit path. Do not
omit `--limit`, `--sample`, or explicit `--questions` for a single-conversation
LoCoMo run.

If the log says `stack V2: pushing ... to one stack`, the run is using the
wrong arm for MemoryKernel evaluation.

Legacy arms such as `vector`, `graph`, `smart`, `smart_unified`, and
`smart_hybrid` are Delve-stack comparison paths. Do not use them for the
current conv-26 MemoryKernel gate.
