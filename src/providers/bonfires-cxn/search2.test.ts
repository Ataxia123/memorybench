import { describe, expect, test } from "bun:test"
import { BonfiresCxnProvider, type CxnDeps } from "./index"
import type { CxnArtifacts } from "./artifacts"
import type { CxnConfig } from "./config"
import type { Aggregates, StatementEntry } from "./retrieval2"
import { VOYAGE_MODEL, type FetchLike } from "./voyage"

// ---------- shapes returned by search() (kept local — search() returns unknown[]) ----------

interface UtteranceItem {
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

interface ContextItem {
  kind: "cxn_context"
  lines: string[]
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
  }
}

function isUtterance(item: unknown): item is UtteranceItem {
  return (item as { kind?: string }).kind === "cxn_utterance"
}
function isContext(item: unknown): item is ContextItem {
  return (item as { kind?: string }).kind === "cxn_context"
}

// ---------- config / deps helpers ----------

function baseConfig(overrides: Partial<CxnConfig> = {}): CxnConfig {
  return {
    neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
    groupId: "g", artifactsDir: "/x", utteranceMapPath: "/x/m.json",
    expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
    voyageApiKey: "key", corpusPath: "/x/corpus.json", sessionTurnsPath: "/x/turns.json",
    embeddingsPath: "/x/emb.json",
    laneP: false, blendDense: 0.7, blendSparse: 0.3, poolK: 40, finalK: 20,
    deltaCxn: 0.3, deltaEp: 0.3, hydrateTop: 5, hydrateWindow: 2,
    ...overrides,
  }
}

// search() must never touch the KG — the whole point of v2 is a live-query
// dense floor over pre-embedded artifacts, no Cypher walk.
function depsWithFetch(fetchImpl: FetchLike): CxnDeps {
  return {
    runCypher: async () => {
      throw new Error("bonfires-cxn v2 search: must not run Cypher")
    },
    fetchImpl,
  }
}

function fixedVectorFetch(vector: number[]): FetchLike {
  return (async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as { input: string[] }
    return new Response(JSON.stringify({ data: body.input.map(() => ({ embedding: vector })) }), { status: 200 })
  }) as FetchLike
}

function countingFixedVectorFetch(vector: number[]): { fetchImpl: FetchLike; calls: () => number } {
  let count = 0
  const fetchImpl = (async (_url: string, init: RequestInit) => {
    count += 1
    const body = JSON.parse(String(init.body)) as { input: string[] }
    return new Response(JSON.stringify({ data: body.input.map(() => ({ embedding: vector })) }), { status: 200 })
  }) as FetchLike
  return { fetchImpl, calls: () => count }
}

function failingFetch(status: number): FetchLike {
  return (async () => new Response("boom", { status })) as FetchLike
}

// ---------- statement fixture helper ----------

function stmt(overrides: Partial<StatementEntry>): StatementEntry {
  return {
    hash: "h", utterance: "u", ts: "2023-05-08T13:00:00Z", actor_id: "A",
    session: "s1", session_index: 0, construct_ids: ["residual.v1"], ...overrides,
  }
}

// 3 statements, distinct 2-d vectors, 1 cxn + 1 episode aggregate, minimal turns —
// used by the floor / laneP / memo / failure tests. Query vector fixed at [1, 0].
// h1 is ask-shaped (construct marker) with no session-mate — replyExpansion is a
// no-op here so it doesn't confound the floor/laneP comparisons.
function fixtureArtifactsA(): CxnArtifacts {
  const statements = new Map<string, StatementEntry>([
    ["h1", stmt({
      hash: "h1", utterance: "Melanie asked Caroline about her day", ts: "2023-05-08T13:00:00Z",
      actor_id: "Melanie", session: "s1", session_index: 0, construct_ids: ["person.ask.v1"],
    })],
    ["h2", stmt({
      hash: "h2", utterance: "The weather was nice today", ts: "2023-05-08T14:00:00Z",
      actor_id: "Caroline", session: "s2", session_index: 0, construct_ids: ["residual.v1"],
    })],
    ["h3", stmt({
      hash: "h3", utterance: "They ate dinner together", ts: "2023-05-08T12:00:00Z",
      actor_id: "Caroline", session: "s3", session_index: 0, construct_ids: ["residual.v1"],
    })],
  ])
  const vectors = new Map<string, number[]>([
    ["h1", [1, 3]],
    ["h2", [1, 0]],
    ["h3", [0, 1]],
  ])
  const turns = new Map([
    ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "Melanie", text: "asked about day", blip_caption: null }]],
    ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "Caroline", text: "nice weather", blip_caption: null }]],
    ["s3", [{ ts: "2023-05-08T12:00:00Z", speaker: "Caroline", text: "ate dinner", blip_caption: null }]],
  ])
  const aggregates: Aggregates = {
    cxn: new Map([["person.ask.v1", [1, 0]]]),
    episode: new Map([["s1", [1, 0]]]),
  }
  return {
    utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
    statements, turns, vectors, aggregates,
  }
}

describe("search v2 — floor (laneP off)", () => {
  test("blended dense-floor scores with lanes metadata + one cxn_context recipe item", async () => {
    const provider = new BonfiresCxnProvider(baseConfig(), fixtureArtifactsA(), depsWithFetch(fixedVectorFetch([1, 0])))
    const results = await provider.search("xyzzy", { containerTag: "t" })

    const utterances = results.filter(isUtterance)
    const contexts = results.filter(isContext)
    expect(utterances.length).toBe(3)
    expect(contexts.length).toBe(1)

    // Display order is chronological by statement ts (h3 12:00 < h1 13:00 < h2 14:00),
    // independent of score/rank.
    expect(utterances.map((u) => u.metadata.utterance_hash)).toEqual(["h3", "h1", "h2"])

    const byHash = new Map(utterances.map((u) => [u.metadata.utterance_hash, u]))
    expect(byHash.get("h3")!.score).toBeCloseTo(0)
    expect(byHash.get("h1")!.score).toBeCloseTo(0.7 * (1 / Math.sqrt(10)), 4)
    expect(byHash.get("h2")!.score).toBeCloseTo(0.7)

    expect(byHash.get("h2")!.metadata.lanes.dense).toBeCloseTo(1)
    expect(byHash.get("h1")!.metadata.lanes.dense).toBeCloseTo(1 / Math.sqrt(10))
    expect(byHash.get("h3")!.metadata.lanes.sparse).toBe(0) // query shares no terms with any statement

    const shas = new Set(utterances.map((u) => u.metadata.query_vector_sha256))
    expect(shas.size).toBe(1) // same query vector -> same sha across all items

    const recipe = contexts[0]!.recipe
    expect(recipe.model).toBe(VOYAGE_MODEL)
    expect(recipe.laneP).toBe(false)
    expect(recipe.blendDense).toBe(0.7)
    expect(recipe.blendSparse).toBe(0.3)
    expect(recipe.poolK).toBe(40)
    expect(recipe.finalK).toBe(20)
    expect(recipe.deltaCxn).toBe(0.3)
    expect(recipe.deltaEp).toBe(0.3)
    expect(recipe.gatesFired).toEqual([])
    expect(recipe.lanePTop).toEqual([])
    expect(contexts[0]!.lines.length).toBeGreaterThan(0)
  })
})

describe("search v2 — lane P", () => {
  test("CXN_LANE_P boosts the aggregate-favored statement past the dense-floor leader", async () => {
    const floorResults = await new BonfiresCxnProvider(
      baseConfig({ laneP: false }), fixtureArtifactsA(), depsWithFetch(fixedVectorFetch([1, 0]))
    ).search("xyzzy", { containerTag: "t" })
    const boostedResults = await new BonfiresCxnProvider(
      baseConfig({ laneP: true }), fixtureArtifactsA(), depsWithFetch(fixedVectorFetch([1, 0]))
    ).search("xyzzy", { containerTag: "t" })

    const scoreOf = (results: unknown[], hash: string) =>
      results.filter(isUtterance).find((u) => u.metadata.utterance_hash === hash)!.score

    // Floor: h2's plain dense match outranks h1.
    expect(scoreOf(floorResults, "h2")).toBeGreaterThan(scoreOf(floorResults, "h1"))
    // With laneP: h1's construct (person.ask.v1) + session (s1) aggregates both
    // cosine-match the query, so its boosted score overtakes h2.
    expect(scoreOf(boostedResults, "h1")).toBeGreaterThan(scoreOf(boostedResults, "h2"))

    const recipe = boostedResults.filter(isContext)[0]!.recipe
    expect(recipe.gatesFired).toContain("laneP")
    expect(recipe.lanePTop.length).toBe(2)
    expect(recipe.lanePTop[0]!.sim).toBeCloseTo(1)
  })
})

describe("search v2 — embedQuery memo", () => {
  test("two identical searches call the voyage fetch once and return byte-identical results", async () => {
    const { fetchImpl, calls } = countingFixedVectorFetch([1, 0])
    const provider = new BonfiresCxnProvider(baseConfig(), fixtureArtifactsA(), depsWithFetch(fetchImpl))
    const a = await provider.search("xyzzy", { containerTag: "t" })
    const b = await provider.search("xyzzy", { containerTag: "t" })
    expect(JSON.stringify(a)).toBe(JSON.stringify(b))
    expect(calls()).toBe(1)
  })
})

describe("search v2 — voyage failure", () => {
  test("non-200 from voyage rejects loudly", async () => {
    const provider = new BonfiresCxnProvider(baseConfig(), fixtureArtifactsA(), depsWithFetch(failingFetch(500)))
    await expect(provider.search("xyzzy", { containerTag: "t" })).rejects.toThrow(/500/)
  })
})

describe("search v2 — reply expansion", () => {
  // poolK=1 forces only the ask-shaped top hit into the initial pool; its
  // session_index+1 reply is otherwise excluded until replyExpansion pulls it in.
  function fixtureArtifactsReply(): CxnArtifacts {
    const statements = new Map<string, StatementEntry>([
      ["ask1", stmt({
        hash: "ask1", utterance: "Melanie asked Caroline about her pets", ts: "2023-05-08T13:00:00Z",
        actor_id: "Melanie", session: "s1", session_index: 3, construct_ids: ["person.ask.v1"],
      })],
      ["reply1", stmt({
        hash: "reply1", utterance: "Caroline said the pets are Luna and Oliver", ts: "2023-05-08T13:01:00Z",
        actor_id: "Caroline", session: "s1", session_index: 4, construct_ids: ["residual.v1"],
      })],
      ["filler1", stmt({
        hash: "filler1", utterance: "They discussed the weather instead", ts: "2023-05-08T13:02:00Z",
        actor_id: "Caroline", session: "s2", session_index: 0, construct_ids: ["residual.v1"],
      })],
    ])
    const vectors = new Map<string, number[]>([
      ["ask1", [1, 0]],
      ["reply1", [0, 1]],
      ["filler1", [-1, 0]],
    ])
    return {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements, turns: new Map(), vectors, aggregates: { cxn: new Map(), episode: new Map() },
    }
  }

  test("ask-shaped top hit pulls its session_index+1 neighbor beyond the pool cutoff", async () => {
    const provider = new BonfiresCxnProvider(
      baseConfig({ poolK: 1, finalK: 5 }), fixtureArtifactsReply(), depsWithFetch(fixedVectorFetch([1, 0]))
    )
    const results = await provider.search("banana", { containerTag: "t" })
    const utterances = results.filter(isUtterance)

    expect(utterances.map((u) => u.metadata.utterance_hash).sort()).toEqual(["ask1", "reply1"])

    const ask = utterances.find((u) => u.metadata.utterance_hash === "ask1")!
    const reply = utterances.find((u) => u.metadata.utterance_hash === "reply1")!
    expect(reply.score).toBeCloseTo(ask.score * 0.8)

    const recipe = results.filter(isContext)[0]!.recipe
    expect(recipe.gatesFired).toEqual(["reply"])
  })
})
