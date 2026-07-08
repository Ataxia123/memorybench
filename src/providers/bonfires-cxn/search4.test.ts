import { describe, expect, test } from "bun:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { loadArtifacts } from "./artifacts"
import { loadCxnConfig, type CxnConfig } from "./config"
import { VOYAGE_MODEL } from "./voyage"

const baseEnv: Record<string, string> = {
  CXN_NEO4J_URI: "bolt://x", CXN_NEO4J_USER: "u", CXN_NEO4J_PASSWORD: "p",
  CXN_GROUP_ID: "g", CXN_ARTIFACTS_DIR: "/tmp", CXN_UTTERANCE_MAP: "/tmp/m.json",
  CXN_VOYAGE_API_KEY: "k", CXN_CORPUS: "/tmp/c.json", CXN_SESSION_TURNS: "/tmp/t.json",
  CXN_EMBEDDINGS: "/tmp/e.json", CXN_Q: "1", CXN_COMPREHEND_URL: "http://127.0.0.1:1",
}

describe("config v4 (leg 5)", () => {
  test("defaults: mmr/captions off, lambda 0.3, damp 1.0", () => {
    const cfg = loadCxnConfig(baseEnv)
    expect(cfg.mmr).toBe(false)
    expect(cfg.captions).toBe(false)
    expect(cfg.mmrLambda).toBe(0.3)
    expect(cfg.captionDamp).toBe(1.0)
    expect(cfg.captionsPath).toBeUndefined()
  })
  test("CXN_CAPTIONS=1 requires CXN_CAPTIONS_PATH", () => {
    expect(() => loadCxnConfig({ ...baseEnv, CXN_CAPTIONS: "1" })).toThrow(/CXN_CAPTIONS_PATH/)
    const cfg = loadCxnConfig({ ...baseEnv, CXN_CAPTIONS: "1", CXN_CAPTIONS_PATH: "/tmp/cap.json" })
    expect(cfg.captions).toBe(true)
    expect(cfg.captionsPath).toBe("/tmp/cap.json")
  })
  test("CXN_MMR=1 without CXN_Q throws loudly", () => {
    const env = { ...baseEnv, CXN_MMR: "1" }
    delete (env as Record<string, string | undefined>).CXN_Q
    delete (env as Record<string, string | undefined>).CXN_COMPREHEND_URL
    expect(() => loadCxnConfig(env)).toThrow(/CXN_MMR.*CXN_Q|CXN_Q.*CXN_MMR/)
    expect(loadCxnConfig({ ...baseEnv, CXN_MMR: "1" }).mmr).toBe(true)
  })
})

// ---------- loader fixture helpers (mirrors artifacts.test.ts) ----------

function writeV2Fixtures(dir: string): { corpusPath: string; sessionTurnsPath: string; embeddingsPath: string } {
  const corpusPath = join(dir, "corpus.json")
  writeFileSync(
    corpusPath,
    JSON.stringify({
      abc123abc123abc1: {
        utterance: "hi",
        ts: "2023-05-08T13:56:00Z",
        actor_id: "Caroline",
        session: "s1",
        session_index: 0,
        construct_ids: ["person.ask.v1"],
      },
    })
  )
  const sessionTurnsPath = join(dir, "turns.json")
  writeFileSync(
    sessionTurnsPath,
    JSON.stringify({
      s1: [{ ts: "2023-05-08T13:56:00Z", speaker: "Caroline", text: "hi", blip_caption: null }],
    })
  )
  const embeddingsPath = join(dir, "embeddings.json")
  writeFileSync(
    embeddingsPath,
    JSON.stringify({
      model: VOYAGE_MODEL,
      dim: 3,
      statements: { abc123abc123abc1: [1, 0, 0] },
      aggregates: { cxn: { "person.ask.v1": [1, 0, 0] }, episode: { s1: [1, 0, 0] } },
    })
  )
  return { corpusPath, sessionTurnsPath, embeddingsPath }
}

function fixtureConfig(overrides: Partial<CxnConfig> = {}): CxnConfig {
  const dir = mkdtempSync(join(tmpdir(), "cxn-search4-"))
  writeFileSync(
    join(dir, "census.json"),
    JSON.stringify({
      per_card: [{ construct_id: "person.ask.v1", bound: 72, suppressed: 0 }],
    })
  )
  writeFileSync(join(dir, "fold_plan.jsonl"), '{"kind":"card"}\n')
  const mapPath = join(dir, "map.json")
  writeFileSync(
    mapPath,
    JSON.stringify({
      abc123abc123abc1: { utterance: "hi", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline", session: "s1" },
    })
  )
  const { corpusPath, sessionTurnsPath, embeddingsPath } = writeV2Fixtures(dir)
  return {
    neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
    groupId: "g", artifactsDir: dir, utteranceMapPath: mapPath,
    expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
    voyageApiKey: "key", corpusPath, sessionTurnsPath, embeddingsPath,
    laneP: false, blendDense: 0.7, blendSparse: 0.3, poolK: 40, finalK: 20,
    deltaCxn: 0.3, deltaEp: 0.3, hydrateTop: 5, hydrateWindow: 2,
    captions: false,
    ...overrides,
  }
}

function writeCaptionsFixture(
  dir: string,
  items: Record<string, { caption: string; ts: string; actor_id: string; session: string; vector: number[] }>,
  overrides: { model?: string; dim?: number } = {}
): string {
  const captionsPath = join(dir, "captions.json")
  writeFileSync(
    captionsPath,
    JSON.stringify({
      model: overrides.model ?? VOYAGE_MODEL,
      dim: overrides.dim ?? 3,
      items,
    })
  )
  return captionsPath
}

describe("captions loader (leg 5)", () => {
  test("happy path: 2 captions load, sorted ids, captionVectors dims validated", async () => {
    const cfg = fixtureConfig()
    const captionsPath = writeCaptionsFixture(cfg.artifactsDir, {
      capB: { caption: "a dog running", ts: "2023-05-08T13:57:00Z", actor_id: "Caroline", session: "s1", vector: [0, 1, 0] },
      capA: { caption: "a cat sleeping", ts: "2023-05-08T13:56:30Z", actor_id: "Melanie", session: "s1", vector: [1, 0, 0] },
    })
    const artifacts = await loadArtifacts({ ...cfg, captions: true, captionsPath })

    expect([...artifacts.captions!.keys()]).toEqual(["capA", "capB"])
    expect(artifacts.captions!.get("capA")).toEqual({
      caption: "a cat sleeping", ts: "2023-05-08T13:56:30Z", actor_id: "Melanie", session: "s1",
    })
    expect(artifacts.captionVectors!.get("capA")).toEqual([1, 0, 0])
    expect(artifacts.captionVectors!.get("capB")).toEqual([0, 1, 0])
    expect([...artifacts.captionVectors!.keys()]).toEqual(["capA", "capB"])
  })

  test("caption dim != main dim throws naming both", async () => {
    const cfg = fixtureConfig()
    const captionsPath = writeCaptionsFixture(
      cfg.artifactsDir,
      { capA: { caption: "a cat sleeping", ts: "2023-05-08T13:56:30Z", actor_id: "Melanie", session: "s1", vector: [1, 0, 0, 0] } },
      { dim: 4 }
    )
    await expect(loadArtifacts({ ...cfg, captions: true, captionsPath })).rejects.toThrow(
      /caption artifact dim 4 != embeddings dim 3/
    )
  })

  test("caption vector wrong length throws naming the caption id", async () => {
    const cfg = fixtureConfig()
    const captionsPath = writeCaptionsFixture(cfg.artifactsDir, {
      capA: { caption: "a cat sleeping", ts: "2023-05-08T13:56:30Z", actor_id: "Melanie", session: "s1", vector: [1, 0] },
    })
    await expect(loadArtifacts({ ...cfg, captions: true, captionsPath })).rejects.toThrow(
      /caption capA.*got length 2, expected 3/
    )
  })

  test("cfg.captions false => captions/captionVectors undefined even if path set", async () => {
    const cfg = fixtureConfig()
    const captionsPath = writeCaptionsFixture(cfg.artifactsDir, {
      capA: { caption: "a cat sleeping", ts: "2023-05-08T13:56:30Z", actor_id: "Melanie", session: "s1", vector: [1, 0, 0] },
    })
    const artifacts = await loadArtifacts({ ...cfg, captions: false, captionsPath })
    expect(artifacts.captions).toBeUndefined()
    expect(artifacts.captionVectors).toBeUndefined()
  })
})
