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

export function loadCxnConfig(env: Record<string, string | undefined> = process.env): CxnConfig {
  return {
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
  }
}
