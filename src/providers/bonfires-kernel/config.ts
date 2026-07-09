export interface KernelConfig {
  apiUrl: string
  bonfireId: string
  apiKey: string
  actorId: string
  expectedCardsDigest?: string
  // Pinned-batches mode: when set, foldIndex() reads this JSON file (an
  // array of message-batch arrays, e.g. the parity corpus's
  // conv26_batches.json) and POSTs it VERBATIM as `message_batches`,
  // bypassing sessionToMessageBatch()'s reassembly entirely. This exists
  // because the parity corpus's extraction is pinned to a content-keyed
  // cache on the kernel side: if the bench reassembles batches itself and
  // that reassembly differs from what produced the pinned cache key (a
  // dropped field, a re-ordered session, a re-derived timestamp), the fold
  // silently misses the cache and re-extracts live against an LLM, drifting
  // the corpus out from under the pinned recipe. Pinning the raw batch JSON
  // is the only byte-faithful guarantee. Default (unset) keeps the normal
  // session -> message-batch assembly path for fresh corpora where no
  // pinned cache key exists yet.
  batchesPath?: string
  // Census-digest tripwire: when set, foldIndex() throws if the fold
  // response's `census_digest` differs from this value — a loud signal
  // that the fold re-extracted instead of hitting the pinned cache.
  expectedCensusDigest?: string
  // Search-only mode (KERNELB_SKIP_FOLD): when true, awaitIndexing() never
  // POSTs /kernel/index at all — no fold runs. This exists for parity/resume
  // runs where the artifact bundle (fold output + pinned normalization cache)
  // was staged out-of-band and the bonfire is already folded; re-running the
  // fold here would be redundant at best and, against a READONLY-pinned
  // cache, would just no-op the fold's cache writes while still burning an
  // LLM extraction pass. The bench still preflights `KERNELB_EXPECTED_CARDS_DIGEST`
  // (if set) so a mismatched pre-folded bonfire fails loudly instead of
  // silently scoring against the wrong artifact. Default false (normal fold
  // path, unchanged).
  skipFold: boolean
}

function required(env: Record<string, string | undefined>, key: string): string {
  const value = env[key]?.trim()
  if (!value) throw new Error(`bonfires-kernel: missing required env ${key}`)
  return value
}

function optionalBool(
  env: Record<string, string | undefined>,
  key: string,
  fallback: boolean
): boolean {
  const raw = env[key]?.trim()
  if (!raw) return fallback
  const lower = raw.toLowerCase()
  if (lower === "1" || lower === "true") return true
  if (lower === "0" || lower === "false") return false
  throw new Error(
    `bonfires-kernel: ${key} must be one of "", "0", "1", "true", "false" (case-insensitive), got "${raw}"`
  )
}

export function loadKernelConfig(
  env: Record<string, string | undefined> = process.env
): KernelConfig {
  return {
    apiUrl: required(env, "KERNELB_API_URL"),
    bonfireId: required(env, "KERNELB_BONFIRE_ID"),
    apiKey: required(env, "KERNELB_API_KEY"),
    actorId: env.KERNELB_ACTOR_ID?.trim() || "bench",
    expectedCardsDigest: env.KERNELB_EXPECTED_CARDS_DIGEST?.trim() || undefined,
    batchesPath: env.KERNELB_BATCHES_PATH?.trim() || undefined,
    expectedCensusDigest: env.KERNELB_EXPECTED_CENSUS_DIGEST?.trim() || undefined,
    skipFold: optionalBool(env, "KERNELB_SKIP_FOLD", false),
  }
}
