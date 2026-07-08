export interface CxnConfig {
  neo4jUri: string
  neo4jUser: string
  neo4jPassword: string
  groupId: string
  artifactsDir: string
  utteranceMapPath: string
  expectedNodes: number
  expectedEdges: number
  topKFirings: number
  maxSeedEntities: number
  // v2 (live-query dense floor + lane P search). Optional at the type level so
  // pre-existing leg-1 fixtures (preflight.test.ts) that build a bare CxnConfig
  // literal keep compiling unchanged; loadCxnConfig() always populates them
  // (env-required), and requireState2() asserts + narrows them at runtime.
  voyageApiKey?: string
  corpusPath?: string
  sessionTurnsPath?: string
  embeddingsPath?: string
  laneP?: boolean
  blendDense?: number
  blendSparse?: number
  poolK?: number
  finalK?: number
  deltaCxn?: number
  deltaEp?: number
  hydrateTop?: number
  hydrateWindow?: number
  // v3 (kernel-native comprehend sidecar + four affordance channels). Optional
  // at the type level for the same reason as the v2 block above: pre-existing
  // leg-1/leg-2 fixtures that build a bare CxnConfig literal keep compiling
  // unchanged; loadCxnConfig() always populates them (env-required iff q=1),
  // and requireState2() defaults them at runtime for the injected-config path.
  comprehendUrl?: string
  q?: boolean
  qStrata?: boolean
  qGates?: boolean
  qSeed?: boolean
  qAnswer?: boolean
  qDelta?: number
  qSeedEntityW?: number
  qSeedVerbW?: number
  qSeedLaneW?: number
}

function required(env: Record<string, string | undefined>, key: string): string {
  const value = env[key]?.trim()
  if (!value) throw new Error(`bonfires-cxn: missing required env ${key}`)
  return value
}

function optionalInt(env: Record<string, string | undefined>, key: string, fallback: number): number {
  const raw = env[key]?.trim()
  if (!raw) return fallback
  const value = Number(raw)
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`bonfires-cxn: ${key} must be a positive integer, got "${raw}"`)
  }
  return value
}

function optionalFloat(env: Record<string, string | undefined>, key: string, fallback: number): number {
  const raw = env[key]?.trim()
  if (!raw) return fallback
  const value = Number(raw)
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`bonfires-cxn: ${key} must be a finite number >= 0, got "${raw}"`)
  }
  return value
}

function optionalBool(env: Record<string, string | undefined>, key: string, fallback: boolean): boolean {
  const raw = env[key]?.trim()
  if (!raw) return fallback
  const lower = raw.toLowerCase()
  if (lower === "1" || lower === "true") return true
  if (lower === "0" || lower === "false") return false
  throw new Error(`bonfires-cxn: ${key} must be one of "", "0", "1", "true", "false" (case-insensitive), got "${raw}"`)
}

export function loadCxnConfig(env: Record<string, string | undefined> = process.env): CxnConfig {
  const config = {
    neo4jUri: required(env, "CXN_NEO4J_URI"),
    neo4jUser: required(env, "CXN_NEO4J_USER"),
    neo4jPassword: required(env, "CXN_NEO4J_PASSWORD"),
    groupId: required(env, "CXN_GROUP_ID"),
    artifactsDir: required(env, "CXN_ARTIFACTS_DIR"),
    utteranceMapPath: required(env, "CXN_UTTERANCE_MAP"),
    expectedNodes: optionalInt(env, "CXN_EXPECTED_NODES", 6419),
    expectedEdges: optionalInt(env, "CXN_EXPECTED_EDGES", 9750),
    topKFirings: optionalInt(env, "CXN_TOP_K_FIRINGS", 20),
    maxSeedEntities: optionalInt(env, "CXN_MAX_SEED_ENTITIES", 12),
    voyageApiKey: required(env, "CXN_VOYAGE_API_KEY"),
    corpusPath: required(env, "CXN_CORPUS"),
    sessionTurnsPath: required(env, "CXN_SESSION_TURNS"),
    embeddingsPath: required(env, "CXN_EMBEDDINGS"),
    laneP: optionalBool(env, "CXN_LANE_P", false),
    blendDense: optionalFloat(env, "CXN_BLEND_DENSE", 0.7),
    blendSparse: optionalFloat(env, "CXN_BLEND_SPARSE", 0.3),
    poolK: optionalInt(env, "CXN_POOL_K", 40),
    finalK: optionalInt(env, "CXN_FINAL_K", 20),
    deltaCxn: optionalFloat(env, "CXN_DELTA_CXN", 0.3),
    deltaEp: optionalFloat(env, "CXN_DELTA_EP", 0.3),
    hydrateTop: optionalInt(env, "CXN_HYDRATE_TOP", 5),
    hydrateWindow: optionalInt(env, "CXN_HYDRATE_WINDOW", 2),
    q: optionalBool(env, "CXN_Q", false),
    comprehendUrl: env.CXN_COMPREHEND_URL?.trim() || undefined,
    qStrata: optionalBool(env, "CXN_Q_STRATA", true),
    qGates: optionalBool(env, "CXN_Q_GATES", true),
    qSeed: optionalBool(env, "CXN_Q_SEED", true),
    qAnswer: optionalBool(env, "CXN_Q_ANSWER", true),
    qDelta: optionalFloat(env, "CXN_QDELTA", 0.3),
    qSeedEntityW: optionalFloat(env, "CXN_QSEED_ENTITY_W", 2.0),
    qSeedVerbW: optionalFloat(env, "CXN_QSEED_VERB_W", 1.0),
    qSeedLaneW: optionalFloat(env, "CXN_QSEED_LANE_W", 0.5),
  }
  if (config.q && !config.comprehendUrl) {
    throw new Error("bonfires-cxn: CXN_Q=1 requires CXN_COMPREHEND_URL")
  }
  return config
}
