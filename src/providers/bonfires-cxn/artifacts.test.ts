import { describe, expect, test } from "bun:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { loadArtifacts } from "./artifacts"
import type { CxnConfig } from "./config"
import { VOYAGE_MODEL } from "./voyage"

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
  const dir = mkdtempSync(join(tmpdir(), "cxn-artifacts-"))
  writeFileSync(
    join(dir, "census.json"),
    JSON.stringify({
      per_card: [
        { construct_id: "person.supported.v1", bound: 44, suppressed: 0 },
        { construct_id: "person.ask.v1", bound: 72, suppressed: 0 },
      ],
    })
  )
  writeFileSync(
    join(dir, "fold_plan.jsonl"),
    '{"kind":"card"}\n{"kind":"residual"}\n{"kind":"residual"}\n'
  )
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
    ...overrides,
  }
}

describe("loadArtifacts", () => {
  test("loads map, entrenchment, and plan count", async () => {
    const artifacts = await loadArtifacts(fixtureConfig())
    expect(artifacts.planRecordCount).toBe(3)
    expect(artifacts.entrenchmentByConstruct.get("person.ask.v1")).toBe(72)
    expect(artifacts.utteranceMap.get("abc123abc123abc1")?.actor_id).toBe("Caroline")
  })

  test("loads v2 statements/turns/vectors/aggregates maps", async () => {
    const artifacts = await loadArtifacts(fixtureConfig())
    expect(artifacts.statements?.get("abc123abc123abc1")?.session_index).toBe(0)
    expect(artifacts.turns?.get("s1")?.[0]?.text).toBe("hi")
    expect(artifacts.vectors?.get("abc123abc123abc1")).toEqual([1, 0, 0])
    expect(artifacts.aggregates?.cxn.get("person.ask.v1")).toEqual([1, 0, 0])
    expect(artifacts.aggregates?.episode.get("s1")).toEqual([1, 0, 0])
  })

  test("throws on embeddings model mismatch", async () => {
    const dir = mkdtempSync(join(tmpdir(), "cxn-artifacts-mismatch-"))
    const cfg = fixtureConfig()
    writeFileSync(
      cfg.embeddingsPath!,
      JSON.stringify({
        model: "some-other-model",
        dim: 3,
        statements: { abc123abc123abc1: [1, 0, 0] },
        aggregates: { cxn: {}, episode: {} },
      })
    )
    void dir
    await expect(loadArtifacts(cfg)).rejects.toThrow(/model mismatch/)
  })

  test("throws when a corpus hash has no embedding vector", async () => {
    const cfg = fixtureConfig()
    writeFileSync(
      cfg.embeddingsPath!,
      JSON.stringify({
        model: VOYAGE_MODEL,
        dim: 3,
        statements: {},
        aggregates: { cxn: {}, episode: {} },
      })
    )
    await expect(loadArtifacts(cfg)).rejects.toThrow(/no embedding vector/)
  })

  test("throws naming the hash and both lengths when a statement vector has the wrong dimension", async () => {
    const cfg = fixtureConfig()
    writeFileSync(
      cfg.embeddingsPath!,
      JSON.stringify({
        model: VOYAGE_MODEL,
        dim: 3,
        statements: { abc123abc123abc1: [1, 0] },
        aggregates: { cxn: { "person.ask.v1": [1, 0, 0] }, episode: { s1: [1, 0, 0] } },
      })
    )
    await expect(loadArtifacts(cfg)).rejects.toThrow(
      /statement abc123abc123abc1.*got length 2, expected 3/
    )
  })

  test("throws naming the id and both lengths when a cxn aggregate vector has the wrong dimension", async () => {
    const cfg = fixtureConfig()
    writeFileSync(
      cfg.embeddingsPath!,
      JSON.stringify({
        model: VOYAGE_MODEL,
        dim: 3,
        statements: { abc123abc123abc1: [1, 0, 0] },
        aggregates: { cxn: { "person.ask.v1": [1, 0] }, episode: { s1: [1, 0, 0] } },
      })
    )
    await expect(loadArtifacts(cfg)).rejects.toThrow(
      /cxn aggregate person\.ask\.v1.*got length 2, expected 3/
    )
  })

  test("throws naming the id and both lengths when an episode aggregate vector has the wrong dimension", async () => {
    const cfg = fixtureConfig()
    writeFileSync(
      cfg.embeddingsPath!,
      JSON.stringify({
        model: VOYAGE_MODEL,
        dim: 3,
        statements: { abc123abc123abc1: [1, 0, 0] },
        aggregates: { cxn: { "person.ask.v1": [1, 0, 0] }, episode: { s1: [1, 0] } },
      })
    )
    await expect(loadArtifacts(cfg)).rejects.toThrow(
      /episode aggregate s1.*got length 2, expected 3/
    )
  })

  test("existing dim-3 fixture keeps loading unchanged and exposes dim", async () => {
    const artifacts = await loadArtifacts(fixtureConfig())
    expect(artifacts.dim).toBe(3)
  })
})
