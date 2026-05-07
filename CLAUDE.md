# Memorybench — Running LoCoMo Benches

## Required env (the gotcha)

The bench's answer + judge phases call OpenAI-compatible chat completions. We point them at OpenRouter, NOT OpenAI directly:

```bash
OPENAI_API_KEY="$OPENROUTER_API_KEY"            # OpenRouter key, named OPENAI_API_KEY
OPENAI_BASE_URL="https://openrouter.ai/api/v1"  # OpenAI client → OpenRouter
```

Without these the answer phase fails with `Incorrect API key provided: ''`. Search phase still runs (it hits the local delve server), but every question's answer fails until you set them and resume.

`OPENROUTER_API_KEY` is set in the running delve server's process env (visible via `tr '\0' '\n' < /proc/<pid>/environ`).

## Bench-side env (for the bonfires provider)

```bash
BONFIRES_BONFIRE_ID="358c982d43841eaed6279261"  # The LoCoMo conv-26 bonfire
BONFIRES_API_URL="http://localhost:8000"
BONFIRES_API_KEY="local-dev-api-key"
BONFIRES_ARM="smart"                             # smart = facts + entities + chunks (the default mix)
```

### Image caption ingest (Z9-A)

```bash
LOCOMO_INCLUDE_IMAGE_CAPTIONS=1   # Default OFF — opt in to inject blip_caption messages
```

When set to `1`, `extractSessions` emits a second `UnifiedMessage` after each text turn that has a `blip_caption` field in the LoCoMo JSON. The injected message has:

- `content`: `[image] <blip_caption> (intent: <query>)` (the `(intent: ...)` part is omitted when `query` is absent)
- `metadata.kind`: `"image_caption"`
- `metadata.parent_dia_id`: the source dialogue turn's `dia_id`
- `metadata.img_url`: the source turn's `img_url` array (empty array if absent)

20.8% of LoCoMo messages (1226/5882) carry captions. This is the Z9-A unlock for image-mediated adversarial questions. **Leave unset for baselines** — existing bench runs were ingested without captions and scores are not comparable across the toggle.

## Server-side env (set on the delve server, NOT on the bench)

The chunks_search aux lanes / gates are gated by env vars on the **delve server**, not on bun. To change them you have to restart delve:

| Knob | Default | Notes |
|---|---|---|
| `CHUNKS_AUX_RANKINGS_ENABLED` | `1` | Master toggle for aux lanes |
| `CHUNKS_AUX_KERNEL_ENABLED` | `1` | v24 kernel kill-switch |
| `CHUNKS_NGRAM_CENTROID_ENABLED` | `0` | Repeating-ngram-centroid lane (RRF input) |
| `CHUNKS_NGRAM_GATE_ENABLED` | `0` | Top-K-centroid candidate-pool gate |
| `CHUNKS_COMMUNITY_RRF_ENABLED` | `0` | community-cos lane |
| `CHUNKS_COMMUNITY_GATE_ENABLED` | `0` | community gate |
| `CHUNKS_CASCADE_GATE_ENABLED` | `0` | label-gate (HyperMem-lite) |
| `CHUNKS_ENGRAM_SUBSET_ENABLED` | `1` | engram subset gate (mode=cascade) |
| `CHUNKS_SEARCH_TIMING` | `0` | Per-stage timing log |

To restart delve with a different knob set:
```bash
# Snapshot existing env (keep MONGO/NEO4J/OPENROUTER/etc):
tr '\0' '\n' < /proc/<delve-pid>/environ \
  | grep -E "^(MONGO|NEO4J|REDIS|WEAVIATE|OPENROUTER|BONFIRES|STACK_V2|CHUNKS_|TASK_QUEUE|LOG_LEVEL|ENVIRONMENT|V2_)" \
  > /tmp/delve.env

# Edit /tmp/delve.env, then:
kill <delve-pid>
set -a; source /tmp/delve.env; set +a
nohup uvicorn server:app --host 0.0.0.0 --port 8000 --app-dir src \
  > /tmp/delve.log 2>&1 &
disown
```

## Standard bench invocation

```bash
cd /home/at0x/Vaults/Bonfires/memorybench
OPENAI_API_KEY="$OPENROUTER_API_KEY" \
OPENAI_BASE_URL="https://openrouter.ai/api/v1" \
BONFIRES_BONFIRE_ID="358c982d43841eaed6279261" \
BONFIRES_API_URL="http://localhost:8000" \
BONFIRES_API_KEY="local-dev-api-key" \
BONFIRES_ARM="smart" \
bun run src/index.ts run \
  --provider bonfires --benchmark locomo \
  --run-id "<descriptive-id>" \
  --limit 199 \
  --concurrency 5 \
  --force
```

`--concurrency 5` is the default the previous successful runs used. **A/B comparisons must use the same concurrency** — queries serialize on the single-threaded delve server and per-search latency rises 3–5× under contention. Sequential vs concurrent reports give misleadingly different "search latency" numbers (apparent 3× regression that's purely a measurement artifact).

## Resuming after failure

`--run-id` resumes from the checkpoint. Drop `--force` and don't change `--limit` — the orchestrator picks up where it died:

```bash
bun run src/index.ts run --provider bonfires --benchmark locomo \
  --run-id <same-id> --concurrency 5
```

## Output locations

- Per-question results: `data/runs/<run-id>/results/conv-*.json`
- Checkpoint (resume state): `data/runs/<run-id>/checkpoint.json`
- Aggregate report: `data/runs/<run-id>/report.json`
