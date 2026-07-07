import { describe, expect, test } from "bun:test"
import { loadCxnConfig } from "./config"

const FULL_ENV = {
  CXN_NEO4J_URI: "bolt://localhost:7687",
  CXN_NEO4J_USER: "neo4j",
  CXN_NEO4J_PASSWORD: "pw",
  CXN_GROUP_ID: "6a4d1cb5ad3586be5a708c86",
  CXN_ARTIFACTS_DIR: "/tmp/artifacts",
  CXN_UTTERANCE_MAP: "/tmp/map.json",
}

describe("loadCxnConfig", () => {
  test("loads full config with defaults for the four optional knobs", () => {
    const cfg = loadCxnConfig(FULL_ENV)
    expect(cfg.groupId).toBe("6a4d1cb5ad3586be5a708c86")
    expect(cfg.expectedNodes).toBe(6419)
    expect(cfg.expectedEdges).toBe(9750)
    expect(cfg.topKFirings).toBe(20)
    expect(cfg.maxSeedEntities).toBe(12)
  })

  test("throws naming the missing required var", () => {
    const { CXN_GROUP_ID: _omit, ...rest } = FULL_ENV
    expect(() => loadCxnConfig(rest)).toThrow(/CXN_GROUP_ID/)
  })

  test("rejects non-positive-integer overrides", () => {
    expect(() => loadCxnConfig({ ...FULL_ENV, CXN_TOP_K_FIRINGS: "0" })).toThrow(/CXN_TOP_K_FIRINGS/)
  })
})
