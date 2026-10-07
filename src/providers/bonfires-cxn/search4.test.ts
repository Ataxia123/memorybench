import { describe, expect, test } from "bun:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { loadArtifacts, type CxnArtifacts } from "./artifacts"
import { loadCxnConfig, type CxnConfig } from "./config"
import { BonfiresCxnProvider, type CxnDeps } from "./index"
import type { Comprehension } from "./affordance"
import type { StatementEntry } from "./retrieval2"
import type { FetchLike } from "./voyage"
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
  test("CXN_CAPTIONS=1 without CXN_Q throws loudly", () => {
    const env = { ...baseEnv, CXN_CAPTIONS: "1", CXN_CAPTIONS_PATH: "/tmp/cap.json" }
    delete (env as Record<string, string | undefined>).CXN_Q
    delete (env as Record<string, string | undefined>).CXN_COMPREHEND_URL
    expect(() => loadCxnConfig(env)).toThrow(/CXN_CAPTIONS.*CXN_Q|CXN_Q.*CXN_CAPTIONS/)
    expect(loadCxnConfig({ ...baseEnv, CXN_CAPTIONS: "1", CXN_CAPTIONS_PATH: "/tmp/cap.json" }).captions).toBe(true)
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

  test("caption id colliding with an existing statement hash throws naming the id", async () => {
    const cfg = fixtureConfig()
    // "abc123abc123abc1" is the statement hash minted by writeV2Fixtures above.
    const captionsPath = writeCaptionsFixture(cfg.artifactsDir, {
      abc123abc123abc1: { caption: "a cat sleeping", ts: "2023-05-08T13:56:30Z", actor_id: "Melanie", session: "s1", vector: [1, 0, 0] },
    })
    await expect(loadArtifacts({ ...cfg, captions: true, captionsPath })).rejects.toThrow(
      /caption id abc123abc123abc1 collides with a statement hash/
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

// ---------- search v4 — leg 5 lanes (caption lane + MMR gate) ----------
//
// Mirrors search3.test.ts's constructor-injection pattern (that file exports
// nothing, so the small pieces below are local re-derivations, not imports):
// inject cfg + artifacts (now including captions/captionVectors) + a fake
// fetch that routes voyage vs comprehend calls by URL.

interface StatementItem {
  kind: "cxn_utterance"
  text: string
  score: number
  metadata: {
    utterance_hash: string
    construct_ids: string[]
    session: string
    lanes: { dense: number; sparse: number }
    query_vector_sha256: string
  }
}

interface CaptionItem {
  kind: "cxn_utterance"
  text: string
  score: number
  metadata: {
    caption_id: string
    lanes: { dense: number; sparse: number }
  }
}

interface ContextItemV4 {
  kind: "cxn_context"
  lines: string[]
  directive: string | null
  recipe: {
    model: string
    laneP: boolean
    blendDense: number
    blendSparse: number
    poolK: number
    finalK: number
    deltaCxn: number
    deltaEp: number
    gatesFired: string[]
    lanePTop: Array<{ key: string; sim: number }>
    q?: boolean
    qStrata?: boolean
    qGates?: boolean
    qSeed?: boolean
    qAnswer?: boolean
    qDelta?: number
    qSeedEntityW?: number
    qSeedVerbW?: number
    qSeedLaneW?: number
    directiveVersion?: number
    affordancesFired?: string[]
    fallback?: boolean
    comprehend?: { probe: string; matched_cxn_ids: string[]; wh_slot: string | null } | null
    mmr?: { enabled: boolean; lambda: number; pool?: string[] }
    captions?: { loaded: number; inPool: number; inFinalK: number; damp: number }
  }
}

function isStatementItem(item: unknown): item is StatementItem {
  if (!item || typeof item !== "object") return false
  const record = item as { kind?: unknown; metadata?: { utterance_hash?: unknown } }
  return record.kind === "cxn_utterance" && typeof record.metadata?.utterance_hash === "string"
}
function isCaptionItem(item: unknown): item is CaptionItem {
  if (!item || typeof item !== "object") return false
  const record = item as { kind?: unknown; metadata?: { caption_id?: unknown } }
  return record.kind === "cxn_utterance" && typeof record.metadata?.caption_id === "string"
}
function isContextV4(item: unknown): item is ContextItemV4 {
  return (item as { kind?: string }).kind === "cxn_context"
}

function baseConfigV4(overrides: Partial<CxnConfig> = {}): CxnConfig {
  return {
    neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
    groupId: "g", artifactsDir: "/x", utteranceMapPath: "/x/m.json",
    expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
    voyageApiKey: "key", corpusPath: "/x/corpus.json", sessionTurnsPath: "/x/turns.json",
    embeddingsPath: "/x/emb.json",
    laneP: false, blendDense: 0.7, blendSparse: 0.3, poolK: 40, finalK: 20,
    deltaCxn: 0.3, deltaEp: 0.3, hydrateTop: 5, hydrateWindow: 2,
    q: false, comprehendUrl: undefined,
    qStrata: true, qGates: true, qSeed: true, qAnswer: true,
    qDelta: 0.3, qSeedEntityW: 2.0, qSeedVerbW: 1.0, qSeedLaneW: 0.5,
    mmr: false, mmrLambda: 0.3, captions: false, captionDamp: 1.0,
    ...overrides,
  }
}

// v4 search() must never touch the KG, same invariant as v2/v3.
function depsWithFetchV4(fetchImpl: FetchLike): CxnDeps {
  return {
    runCypher: async () => {
      throw new Error("bonfires-cxn v4 search: must not run Cypher")
    },
    fetchImpl,
  }
}

const EMPTY_COMPREHENSION_V4: Comprehension = {
  probe: "", matched_cxn_ids: [], bound_cxn_ids: [],
  operators: { neg: false, modal: null }, wh_slot: null, fillers: [], date_fillers: [],
}

function routingFetchV4(opts: { vector: number[]; comprehension?: Comprehension }): FetchLike {
  return (async (url: string, init: RequestInit) => {
    if (url.includes("voyageai.com")) {
      const body = JSON.parse(String(init.body)) as { input: string[] }
      return new Response(JSON.stringify({ data: body.input.map(() => ({ embedding: opts.vector })) }), { status: 200 })
    }
    if (url.endsWith("/comprehend")) {
      return new Response(JSON.stringify(opts.comprehension ?? EMPTY_COMPREHENSION_V4), { status: 200 })
    }
    if (url.endsWith("/health")) {
      return new Response(JSON.stringify({ cards: 1, digest: "d0", construct_ids: [] }), { status: 200 })
    }
    throw new Error(`routingFetchV4: unexpected url ${url}`)
  }) as FetchLike
}

function stmtV4(overrides: Partial<StatementEntry>): StatementEntry {
  return {
    hash: "h", utterance: "u", ts: "2023-05-08T13:00:00Z", actor_id: "A",
    session: "s1", session_index: 0, construct_ids: ["residual.v1"], ...overrides,
  }
}

describe("search v4 — leg 5 lanes (caption lane + MMR gate)", () => {
  // 2 plain statements, no shared BM25 tokens with any query used below —
  // the same shape as search3.test.ts's fixtureParity, re-derived locally.
  function fixtureParityV4(): CxnArtifacts {
    const statements = new Map<string, StatementEntry>([
      ["p1", stmtV4({
        hash: "p1", utterance: "The weather was nice today", ts: "2023-05-08T13:00:00Z",
        actor_id: "A", session: "s1", session_index: 0,
      })],
      ["p2", stmtV4({
        hash: "p2", utterance: "They ate dinner together", ts: "2023-05-08T14:00:00Z",
        actor_id: "B", session: "s2", session_index: 0,
      })],
    ])
    const vectors = new Map<string, number[]>([["p1", [1, 0]], ["p2", [0, 1]]])
    const turns = new Map([
      ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
      ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
    ])
    return {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements, turns, vectors, aggregates: { cxn: new Map(), episode: new Map() },
    }
  }

  // The bare leg-4 shape (search3.test.ts's baseConfigV3, no v4 fields at
  // all — mmr/mmrLambda/captions/captionDamp keys entirely absent from the
  // literal) used as the control side of scenarios 1 and 7.
  function legacyV3Config(overrides: Partial<CxnConfig> = {}): CxnConfig {
    return {
      neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
      groupId: "g", artifactsDir: "/x", utteranceMapPath: "/x/m.json",
      expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
      voyageApiKey: "key", corpusPath: "/x/corpus.json", sessionTurnsPath: "/x/turns.json",
      embeddingsPath: "/x/emb.json",
      laneP: false, blendDense: 0.7, blendSparse: 0.3, poolK: 40, finalK: 20,
      deltaCxn: 0.3, deltaEp: 0.3, hydrateTop: 5, hydrateWindow: 2,
      q: false, comprehendUrl: undefined,
      qStrata: true, qGates: true, qSeed: true, qAnswer: true,
      qDelta: 0.3, qSeedEntityW: 2.0, qSeedVerbW: 1.0, qSeedLaneW: 0.5,
      ...overrides,
    }
  }

  test("1. control parity: mmr:false, captions:false -> byte-identical to a bare leg-4 run; recipe carries no mmr/captions keys", async () => {
    const artifacts = fixtureParityV4()
    const fetchImpl = routingFetchV4({ vector: [1, 0] })

    const legacyCfg = legacyV3Config({ q: true, comprehendUrl: "http://sidecar.local" })
    const v4OffCfg = baseConfigV4({ q: true, comprehendUrl: "http://sidecar.local", mmr: false, captions: false })

    const legacyResult = await new BonfiresCxnProvider(legacyCfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )
    const v4Result = await new BonfiresCxnProvider(v4OffCfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )

    expect(JSON.stringify(v4Result)).toBe(JSON.stringify(legacyResult))
    const recipe = v4Result.filter(isContextV4)[0]!.recipe
    expect("mmr" in recipe).toBe(false)
    expect("captions" in recipe).toBe(false)
  })

  test("2. caption lane: matching caption enters pool+final, renders (image), caption_id set / utterance_hash absent, recipe.captions counts, b:captions fired", async () => {
    // st1/st2 share no BM25 token with "dog" and point away from the query
    // vector; cap1's vector matches the query exactly (dense cos=1) and its
    // caption text contains "dog" (BM25 hit) — both lanes pull it in.
    const artifacts: CxnArtifacts = {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements: new Map<string, StatementEntry>([
        ["st1", stmtV4({ hash: "st1", utterance: "They walked along the beach", session: "s1", session_index: 0 })],
        ["st2", stmtV4({ hash: "st2", utterance: "She read a long novel", session: "s2", session_index: 0 })],
      ]),
      turns: new Map([
        ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
        ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
      ]),
      vectors: new Map([["st1", [0, 1]], ["st2", [0, 1]]]),
      aggregates: { cxn: new Map(), episode: new Map() },
      captions: new Map([["cap1", { caption: "a dog running", ts: "2023-05-08T13:30:00Z", actor_id: "A", session: "s1" }]]),
      captionVectors: new Map([["cap1", [1, 0]]]),
    }
    const fetchImpl = routingFetchV4({ vector: [1, 0], comprehension: EMPTY_COMPREHENSION_V4 })
    const cfg = baseConfigV4({
      q: true, comprehendUrl: "http://sidecar.local", captions: true,
      qStrata: false, qGates: false, qSeed: false, qAnswer: false,
    })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "dog", { containerTag: "t" }
    )

    const captionItem = results.find(isCaptionItem)
    expect(captionItem).toBeDefined()
    expect(captionItem!.text).toContain("(image)")
    expect(captionItem!.text).toContain("a dog running")
    expect(captionItem!.metadata.caption_id).toBe("cap1")
    expect((captionItem!.metadata as Record<string, unknown>).utterance_hash).toBeUndefined()
    // cap1's vector [1,0] is identical to the query vector [1,0] -> cos=1.
    expect(captionItem!.metadata.lanes.dense).toBeCloseTo(1, 5)

    const recipe = results.filter(isContextV4)[0]!.recipe
    expect(recipe.captions).toEqual({ loaded: 1, inPool: 1, inFinalK: 1, damp: 1 })
    expect(recipe.captions!.damp).toBe(1)
    expect(recipe.affordancesFired).toContain("b:captions")
  })

  test("3. caption isolation: statement lanes.sparse identical with captions on vs off (separate-index proof)", async () => {
    const artifacts: CxnArtifacts = {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements: new Map<string, StatementEntry>([
        ["st1", stmtV4({ hash: "st1", utterance: "Melanie loves hiking trips", session: "s1", session_index: 0 })],
        ["st2", stmtV4({ hash: "st2", utterance: "Caroline enjoys quiet mornings", session: "s2", session_index: 0 })],
      ]),
      turns: new Map([
        ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
        ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
      ]),
      vectors: new Map([["st1", [1, 0]], ["st2", [0, 1]]]),
      aggregates: { cxn: new Map(), episode: new Map() },
      captions: new Map([["cap1", { caption: "hiking trail photo", ts: "2023-05-08T13:15:00Z", actor_id: "A", session: "s1" }]]),
      captionVectors: new Map([["cap1", [1, 0]]]),
    }

    const runWith = async (captions: boolean) => {
      const fetchImpl = routingFetchV4({ vector: [1, 0], comprehension: EMPTY_COMPREHENSION_V4 })
      const cfg = baseConfigV4({
        q: true, comprehendUrl: "http://sidecar.local", captions,
        qStrata: false, qGates: false, qSeed: false, qAnswer: false,
      })
      return new BonfiresCxnProvider(cfg, artifacts, depsWithFetchV4(fetchImpl)).search("hiking", { containerTag: "t" })
    }
    const withoutCaptions = await runWith(false)
    const withCaptions = await runWith(true)

    const sparseOf = (results: unknown[], hash: string) =>
      results.filter(isStatementItem).find((s) => s.metadata.utterance_hash === hash)!.metadata.lanes.sparse

    expect(sparseOf(withCaptions, "st1")).toBe(sparseOf(withoutCaptions, "st1"))
    expect(sparseOf(withCaptions, "st2")).toBe(sparseOf(withoutCaptions, "st2"))
    // "hiking" is BM25-rare (only st1 contains it) — a nonzero value here
    // proves the assertion above isn't vacuously comparing two zeros.
    expect(sparseOf(withCaptions, "st1")).toBeGreaterThan(0)
  })

  test("4. caption inertness: qStrata's construct-match boost never reaches a caption score", async () => {
    // Reuses search3.test.ts's fixtureStrata numbers exactly (st1 matches
    // the construct, st2 doesn't; qDelta=1.0 makes the arithmetic exact),
    // plus one caption whose vector matches the query (dense forced to 1 by
    // minMax's single-entry rule, same as st1/st2's floor before strata).
    const artifacts: CxnArtifacts = {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements: new Map<string, StatementEntry>([
        ["st1", stmtV4({
          hash: "st1", utterance: "Someone painted a colorful mural", session: "s1", session_index: 0,
          construct_ids: ["person.paint.v1"],
        })],
        ["st2", stmtV4({ hash: "st2", utterance: "They watched a movie together", session: "s2", session_index: 0 })],
      ]),
      turns: new Map([
        ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
        ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
      ]),
      vectors: new Map([["st1", [0, 1]], ["st2", [1, 0]]]),
      aggregates: { cxn: new Map(), episode: new Map() },
      captions: new Map([["cap1", { caption: "a quiet park bench", ts: "2023-05-08T13:15:00Z", actor_id: "A", session: "s1" }]]),
      captionVectors: new Map([["cap1", [1, 0]]]),
    }
    const comprehension: Comprehension = { ...EMPTY_COMPREHENSION_V4, matched_cxn_ids: ["person.paint.v1"] }
    const fetchImpl = routingFetchV4({ vector: [1, 0], comprehension })
    const cfg = baseConfigV4({
      q: true, comprehendUrl: "http://sidecar.local", captions: true, qStrata: true, qDelta: 1.0,
      qGates: false, qSeed: false, qAnswer: false,
    })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "banana", { containerTag: "t" }
    )

    const st1 = results.filter(isStatementItem).find((s) => s.metadata.utterance_hash === "st1")!
    const captionItem = results.find(isCaptionItem)!

    // st1: dense=0 (cos([1,0],[0,1])), sparse=0 (no BM25 overlap with
    // "banana") -> blend 0; strata boost = 1.0*1/1 = 1.0 -> score 1.0.
    expect(st1.score).toBeCloseTo(1.0, 5)
    // cap1: dense=1 (cos([1,0],[1,0])), sparse=0 -> blend = 0.7*1+0.3*0 = 0.7,
    // damp=1.0. The caption lane merges into `scores` AFTER strataBoost has
    // already run over the statement-only map, so this is the raw caption
    // blend, untouched by qDelta — the inertness invariant.
    expect(captionItem.score).toBeCloseTo(0.7, 5)

    const recipe = results.filter(isContextV4)[0]!.recipe
    expect(recipe.affordancesFired).toContain("q:strata")
  })

  test("5. temporal reach: a qGates window covering only the caption's ts boosts it via the unified ts map", async () => {
    const artifacts: CxnArtifacts = {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements: new Map<string, StatementEntry>([
        ["juneStmt", stmtV4({ hash: "juneStmt", utterance: "They went somewhere", ts: "2023-06-15T10:00:00Z", session: "s1", session_index: 0 })],
      ]),
      turns: new Map([["s1", [{ ts: "2023-06-15T10:00:00Z", speaker: "A", text: "t1", blip_caption: null }]]]),
      vectors: new Map([["juneStmt", [1, 0]]]),
      aggregates: { cxn: new Map(), episode: new Map() },
      captions: new Map([["capMay", { caption: "a sunny park photo", ts: "2023-05-15T10:00:00Z", actor_id: "A", session: "s1" }]]),
      captionVectors: new Map([["capMay", [0, 1]]]),
    }
    const comprehension: Comprehension = {
      ...EMPTY_COMPREHENSION_V4, date_fillers: [{ text: "May 2023", year: 2023, month: 5 }],
    }
    const fetchImpl = routingFetchV4({ vector: [1, 0], comprehension })
    const cfg = baseConfigV4({
      q: true, comprehendUrl: "http://sidecar.local", captions: true, qGates: true,
      qStrata: false, qSeed: false, qAnswer: false,
    })
    // "pineapple" shares no BM25 token with either the statement or caption text.
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "pineapple", { containerTag: "t" }
    )

    const juneStmt = results.filter(isStatementItem).find((s) => s.metadata.utterance_hash === "juneStmt")!
    const capMay = results.find(isCaptionItem)!

    // Both dense maps have exactly one entry (one statement, one caption), so
    // minMax's range===0 rule forces each to 1 regardless of raw cosine;
    // sparse is 0 for both (no BM25 overlap) -> both blend to 0.7*1+0.3*0=0.7
    // before the temporal gate. juneStmt's ts (June) is outside the May
    // window and stays 0.7; capMay's ts (May) is inside it and gets +0.15 ->
    // 0.85 — proof applyTemporalBoost consults the unified tsById map
    // (statements ∪ captions), not just artifacts.statements.
    expect(juneStmt.score).toBeCloseTo(0.7, 5)
    expect(capMay.score).toBeCloseTo(0.85, 5)

    const recipe = results.filter(isContextV4)[0]!.recipe
    expect(recipe.affordancesFired).toContain("q:temporal")
  })

  test("6. MMR gate: list-shape fires b:mmr and lets a diverse item displace a near-duplicate; non-list-shape leaves score order untouched", async () => {
    // Hand-computed BM25 (K1=1.2, B=0.75; avgLength=(3+2+1)/3=2):
    // idf("apple") = ln(1 + (3-3+0.5)/(3+0.5)) = ln(1.142857) ≈ 0.133531.
    // dup1 (tf=3,L=3): denom=3+1.2*(0.25+0.75*3/2)=4.65, score=idf*3*2.2/4.65 ≈ 0.189557.
    // dup2 (tf=2,L=2): denom=2+1.2*(0.25+0.75*2/2)=3.2,  score=idf*2*2.2/3.2  ≈ 0.183605.
    // div  (tf=1,L=1): denom=1+1.2*(0.25+0.75*1/2)=1.75, score=idf*1*2.2/1.75 ≈ 0.167881.
    // dup1 > dup2 > div. dup1/dup2 share the SAME vector [1,0] (cos=1, a
    // literal duplicate direction); div=[0,1] is orthogonal to both (cos=0).
    const artifacts: CxnArtifacts = {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements: new Map<string, StatementEntry>([
        ["dup1", stmtV4({ hash: "dup1", utterance: "apple apple apple", session: "s1", session_index: 0 })],
        ["dup2", stmtV4({ hash: "dup2", utterance: "apple apple", session: "s2", session_index: 0 })],
        ["div", stmtV4({ hash: "div", utterance: "apple", session: "s3", session_index: 0 })],
      ]),
      turns: new Map([
        ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
        ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
        ["s3", [{ ts: "2023-05-08T15:00:00Z", speaker: "A", text: "t3", blip_caption: null }]],
      ]),
      vectors: new Map([["dup1", [1, 0]], ["dup2", [1, 0]], ["div", [0, 1]]]),
      aggregates: { cxn: new Map(), episode: new Map() },
    }
    const idsOf = (results: unknown[]) => results.filter(isStatementItem).map((s) => s.metadata.utterance_hash)

    const runWith = async (whSlot: string | null, mmr: boolean) => {
      const comprehension: Comprehension = { ...EMPTY_COMPREHENSION_V4, wh_slot: whSlot }
      const fetchImpl = routingFetchV4({ vector: [1, 0], comprehension })
      const cfg = baseConfigV4({
        q: true, comprehendUrl: "http://sidecar.local", mmr, mmrLambda: 1.0,
        blendDense: 0, blendSparse: 1.0, finalK: 2,
        qStrata: false, qGates: false, qSeed: false, qAnswer: false,
      })
      return new BonfiresCxnProvider(cfg, artifacts, depsWithFetchV4(fetchImpl)).search("apple", { containerTag: "t" })
    }

    // --- list-shape: MMR fires ---
    const listResults = await runWith("list", true)
    // With lambda=1.0: after dup1 is picked first (highest relevance), dup2's
    // tradeoff value = relevance_dup2 - 1*cos(dup2,dup1) = relevance_dup2 - 1
    // <= 0 (relevance is minmax'd into [0,1]), while div's value =
    // relevance_div - 1*cos(div,dup1) = relevance_div - 0 >= 0 — so div ALWAYS
    // displaces dup2 for the second slot regardless of the exact BM25
    // fractions above. Checked as a SET, not a sequence: all three fixture
    // statements share stmtV4's default ts, so the rendering pass's
    // chronological sort falls to its hash tie-break ("div" < "dup1"),
    // which is a display-order detail orthogonal to what mmrSelect picked.
    expect(new Set(idsOf(listResults))).toEqual(new Set(["dup1", "div"]))

    const listRecipe = listResults.filter(isContextV4)[0]!.recipe
    expect(listRecipe.affordancesFired).toContain("b:mmr")
    expect(listRecipe.mmr!.pool).toEqual(["div", "dup1", "dup2"])
    for (const id of idsOf(listResults)) expect(listRecipe.mmr!.pool).toContain(id)

    // --- non-list-shape: MMR gate does not fire; ordering matches mmr:false ---
    const whatResults = await runWith("what", true)
    const mmrOffResults = await runWith("what", false)
    expect(idsOf(whatResults)).toEqual(idsOf(mmrOffResults))
    expect(idsOf(whatResults)).toEqual(["dup1", "dup2"])

    const whatRecipe = whatResults.filter(isContextV4)[0]!.recipe
    expect(whatRecipe.affordancesFired ?? []).not.toContain("b:mmr")
    expect(whatRecipe.mmr).toEqual({ enabled: true, lambda: 1.0 })
  })

  test("7. captions:false + mmr:false recipe deep-equals the leg-4 q-on recipe shape (no new keys)", async () => {
    const artifacts = fixtureParityV4()
    const fetchImpl = routingFetchV4({ vector: [1, 0] })
    const legacyCfg = legacyV3Config({ q: true, comprehendUrl: "http://sidecar.local" })
    const v4Cfg = baseConfigV4({ q: true, comprehendUrl: "http://sidecar.local", mmr: false, captions: false })

    const legacyResult = await new BonfiresCxnProvider(legacyCfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )
    const v4Result = await new BonfiresCxnProvider(v4Cfg, artifacts, depsWithFetchV4(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )

    const legacyRecipe = legacyResult.filter(isContextV4)[0]!.recipe
    const v4Recipe = v4Result.filter(isContextV4)[0]!.recipe
    expect(Object.keys(v4Recipe).sort()).toEqual(Object.keys(legacyRecipe).sort())
    expect(v4Recipe).toEqual(legacyRecipe)
  })
})
