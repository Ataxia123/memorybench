import { describe, expect, test } from "bun:test"
import { loadCxnConfig } from "./config"

const FULL_ENV = {
  CXN_NEO4J_URI: "bolt://localhost:7687",
  CXN_NEO4J_USER: "neo4j",
  CXN_NEO4J_PASSWORD: "pw",
  CXN_GROUP_ID: "6a4d1cb5ad3586be5a708c86",
  CXN_ARTIFACTS_DIR: "/tmp/artifacts",
  CXN_UTTERANCE_MAP: "/tmp/map.json",
  CXN_VOYAGE_API_KEY: "voyage-key",
  CXN_CORPUS: "/tmp/corpus.json",
  CXN_SESSION_TURNS: "/tmp/turns.json",
  CXN_EMBEDDINGS: "/tmp/embeddings.json",
}

describe("loadCxnConfig", () => {
  test("loads full config with defaults for the four leg-1 optional knobs", () => {
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

  test("throws naming the missing v2 required var", () => {
    const { CXN_VOYAGE_API_KEY: _omit, ...rest } = FULL_ENV
    expect(() => loadCxnConfig(rest)).toThrow(/CXN_VOYAGE_API_KEY/)
  })

  test("rejects non-positive-integer overrides", () => {
    expect(() => loadCxnConfig({ ...FULL_ENV, CXN_TOP_K_FIRINGS: "0" })).toThrow(/CXN_TOP_K_FIRINGS/)
  })

  test("loads defaults for the nine v2 optional knobs", () => {
    const cfg = loadCxnConfig(FULL_ENV)
    expect(cfg.laneP).toBe(false)
    expect(cfg.blendDense).toBe(0.7)
    expect(cfg.blendSparse).toBe(0.3)
    expect(cfg.poolK).toBe(40)
    expect(cfg.finalK).toBe(20)
    expect(cfg.deltaCxn).toBe(0.3)
    expect(cfg.deltaEp).toBe(0.3)
    expect(cfg.hydrateTop).toBe(5)
    expect(cfg.hydrateWindow).toBe(2)
  })

  test("parses CXN_LANE_P truthy forms as boolean true", () => {
    expect(loadCxnConfig({ ...FULL_ENV, CXN_LANE_P: "1" }).laneP).toBe(true)
    expect(loadCxnConfig({ ...FULL_ENV, CXN_LANE_P: "true" }).laneP).toBe(true)
    expect(loadCxnConfig({ ...FULL_ENV, CXN_LANE_P: "0" }).laneP).toBe(false)
  })

  test("rejects non-finite / negative float overrides", () => {
    expect(() => loadCxnConfig({ ...FULL_ENV, CXN_BLEND_DENSE: "abc" })).toThrow(/CXN_BLEND_DENSE/)
    expect(() => loadCxnConfig({ ...FULL_ENV, CXN_DELTA_CXN: "-0.1" })).toThrow(/CXN_DELTA_CXN/)
  })
})
