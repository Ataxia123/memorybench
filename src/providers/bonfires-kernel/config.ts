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
}

function required(env: Record<string, string | undefined>, key: string): string {
  const value = env[key]?.trim()
  if (!value) throw new Error(`bonfires-kernel: missing required env ${key}`)
  return value
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
  }
}
