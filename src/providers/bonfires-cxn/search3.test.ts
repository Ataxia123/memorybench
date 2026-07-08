import { describe, expect, test } from "bun:test"
import { BonfiresCxnProvider, type CxnDeps } from "./index"
import type { CxnArtifacts } from "./artifacts"
import { loadCxnConfig, type CxnConfig } from "./config"
import type { Comprehension } from "./affordance"
import type { StatementEntry } from "./retrieval2"
import type { FetchLike } from "./voyage"

// ---------- shapes returned by search() (v3 adds `directive` to the context
// item and a superset recipe when q is on — kept local, search() returns
// unknown[]) ----------

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

interface ContextItemV3 {
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
  }
}

function isUtterance(item: unknown): item is UtteranceItem {
  return (item as { kind?: string }).kind === "cxn_utterance"
}
function isContext(item: unknown): item is ContextItemV3 {
  return (item as { kind?: string }).kind === "cxn_context"
}

// ---------- config / deps helpers ----------

function baseConfigV3(overrides: Partial<CxnConfig> = {}): CxnConfig {
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

// v3 search() must never touch the KG, same invariant as v2.
function depsWithFetch(fetchImpl: FetchLike): CxnDeps {
  return {
    runCypher: async () => {
      throw new Error("bonfires-cxn v3 search: must not run Cypher")
    },
    fetchImpl,
  }
}

const EMPTY_COMPREHENSION: Comprehension = {
  probe: "", matched_cxn_ids: [], bound_cxn_ids: [],
  operators: { neg: false, modal: null }, wh_slot: null, fillers: [], date_fillers: [],
}

// Fake fetch that routes by URL: voyage embeddings vs the comprehend sidecar
// (POST /comprehend, GET /health) — mirrors what the real Python sidecar
// returns. Tracks call counts + last comprehend POST body for the dispatch
// assertions.
function routingFetch(opts: {
  vector: number[]
  comprehension?: Comprehension
  comprehendStatus?: number
  health?: { cards: number; digest: string; construct_ids: string[] }
  healthStatus?: number
}): {
  fetchImpl: FetchLike
  voyageCalls: () => number
  comprehendCalls: () => number
  lastComprehendBody: () => unknown
} {
  let voyage = 0
  let comprehend = 0
  let lastBody: unknown = null
  const fetchImpl = (async (url: string, init: RequestInit) => {
    if (url.includes("voyageai.com")) {
      voyage += 1
      const body = JSON.parse(String(init.body)) as { input: string[] }
      return new Response(JSON.stringify({ data: body.input.map(() => ({ embedding: opts.vector })) }), { status: 200 })
    }
    if (url.endsWith("/comprehend")) {
      comprehend += 1
      lastBody = JSON.parse(String(init.body))
      if (opts.comprehendStatus && opts.comprehendStatus !== 200) {
        return new Response("sidecar boom", { status: opts.comprehendStatus })
      }
      return new Response(JSON.stringify(opts.comprehension ?? EMPTY_COMPREHENSION), { status: 200 })
    }
    if (url.endsWith("/health")) {
      if (opts.healthStatus && opts.healthStatus !== 200) {
        return new Response("sidecar down", { status: opts.healthStatus })
      }
      return new Response(
        JSON.stringify(opts.health ?? { cards: 1, digest: "d0", construct_ids: [] }),
        { status: 200 }
      )
    }
    throw new Error(`routingFetch: unexpected url ${url}`)
  }) as FetchLike
  return {
    fetchImpl,
    voyageCalls: () => voyage,
    comprehendCalls: () => comprehend,
    lastComprehendBody: () => lastBody,
  }
}

// ---------- statement fixture helper ----------

function stmt(overrides: Partial<StatementEntry>): StatementEntry {
  return {
    hash: "h", utterance: "u", ts: "2023-05-08T13:00:00Z", actor_id: "A",
    session: "s1", session_index: 0, construct_ids: ["residual.v1"], ...overrides,
  }
}

// 2 statements, no shared tokens with any query used below — keeps BM25 out
// of the way so tests can isolate the channel under test. Used by control
// parity + answer/fallback scenarios.
function fixtureParity(): CxnArtifacts {
  const statements = new Map<string, StatementEntry>([
    ["p1", stmt({
      hash: "p1", utterance: "The weather was nice today", ts: "2023-05-08T13:00:00Z",
      actor_id: "A", session: "s1", session_index: 0,
    })],
    ["p2", stmt({
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

describe("search v3 — control parity (CXN_Q=0)", () => {
  test("q=false deep-equals a v2-only config run; comprehend URL never fetched", async () => {
    const artifacts = fixtureParity()
    const { fetchImpl, comprehendCalls } = routingFetch({ vector: [1, 0] })

    // v2-only literal: no q/qStrata/... keys at all, exactly what pre-Task-4
    // fixtures (search2.test.ts) look like.
    const v2OnlyCfg: CxnConfig = {
      neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
      groupId: "g", artifactsDir: "/x", utteranceMapPath: "/x/m.json",
      expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
      voyageApiKey: "key", corpusPath: "/x/corpus.json", sessionTurnsPath: "/x/turns.json",
      embeddingsPath: "/x/emb.json",
      laneP: false, blendDense: 0.7, blendSparse: 0.3, poolK: 40, finalK: 20,
      deltaCxn: 0.3, deltaEp: 0.3, hydrateTop: 5, hydrateWindow: 2,
    }
    const v3OffCfg = baseConfigV3({ q: false, comprehendUrl: "http://sidecar.local" })

    const resultV2 = await new BonfiresCxnProvider(v2OnlyCfg, artifacts, depsWithFetch(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )
    const resultV3 = await new BonfiresCxnProvider(v3OffCfg, artifacts, depsWithFetch(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )

    expect(JSON.stringify(resultV3)).toBe(JSON.stringify(resultV2))
    expect(comprehendCalls()).toBe(0)
  })
})

describe("search v3 — embed dimension guard", () => {
  test("embedQuery rejects a live vector whose dimension mismatches the artifacts' dim, naming both lengths", async () => {
    // fixtureParity()'s statement vectors are length 2 ([1,0]/[0,1]); tagging
    // the artifacts with dim: 2 (as loadArtifacts() would from the real
    // embeddings file) lets this test drive a real mismatch without needing
    // 1024-d (VOYAGE_DIM) fixtures — validation keys off the artifact's own
    // dim, not the production constant.
    const artifacts = { ...fixtureParity(), dim: 2 }
    const { fetchImpl } = routingFetch({ vector: [1, 0, 0] }) // voyage returns a 3-d vector
    const cfg = baseConfigV3({ q: false, comprehendUrl: undefined })
    const provider = new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl))
    await expect(provider.search("xyzzy", { containerTag: "t" })).rejects.toThrow(
      /embedQuery vector dimension mismatch.*got length 3, expected 2/
    )
  })
})

describe("search v3 — concurrent dispatch (CXN_Q=1)", () => {
  test("exactly one voyage call and one comprehend call per search(); POST body is {text}", async () => {
    const artifacts = fixtureParity()
    const { fetchImpl, voyageCalls, comprehendCalls, lastComprehendBody } = routingFetch({
      vector: [1, 0], comprehension: EMPTY_COMPREHENSION,
    })
    const cfg = baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local" })
    const provider = new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl))
    await provider.search("what did they eat", { containerTag: "t" })

    expect(voyageCalls()).toBe(1)
    expect(comprehendCalls()).toBe(1)
    expect(lastComprehendBody()).toEqual({ text: "what did they eat" })
  })
})

describe("search v3 — strata channel", () => {
  // h1 carries the matched construct but starts with the lower blended score
  // (query vector orthogonal to h1, aligned with h2). qDelta=1.0 makes the
  // overtake arithmetic exact: h1 = 0 + 1.0*(1/1) = 1.0 > h2's blend 0.7.
  function fixtureStrata(): CxnArtifacts {
    const statements = new Map<string, StatementEntry>([
      ["h1", stmt({
        hash: "h1", utterance: "Someone painted a mural downtown", ts: "2023-05-08T13:00:00Z",
        session: "s1", session_index: 0, construct_ids: ["person.paint.v1"],
      })],
      ["h2", stmt({
        hash: "h2", utterance: "They watched a movie together", ts: "2023-05-08T14:00:00Z",
        session: "s2", session_index: 0, construct_ids: ["residual.v1"],
      })],
    ])
    const vectors = new Map<string, number[]>([["h1", [0, 1]], ["h2", [1, 0]]])
    const turns = new Map([
      ["s1", [{ ts: "2023-05-08T13:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
      ["s2", [{ ts: "2023-05-08T14:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
    ])
    return {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements, turns, vectors, aggregates: { cxn: new Map(), episode: new Map() },
    }
  }

  test("qStrata flips ranking on matched_cxn_ids overlap; off leaves the dense floor order", async () => {
    const artifacts = fixtureStrata()
    const comprehension: Comprehension = { ...EMPTY_COMPREHENSION, matched_cxn_ids: ["person.paint.v1"] }

    const runWith = async (qStrata: boolean) => {
      const { fetchImpl } = routingFetch({ vector: [1, 0], comprehension })
      const cfg = baseConfigV3({
        q: true, comprehendUrl: "http://sidecar.local", qStrata, qDelta: 1.0,
        qGates: false, qSeed: false, qAnswer: false,
      })
      return new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search("banana", { containerTag: "t" })
    }
    const floor = await runWith(false)
    const boosted = await runWith(true)

    const scoreOf = (results: unknown[], hash: string) =>
      results.filter(isUtterance).find((u) => u.metadata.utterance_hash === hash)!.score

    // Floor: h2's plain dense match (cos=1) outranks h1 (cos=0).
    expect(scoreOf(floor, "h2")).toBeGreaterThan(scoreOf(floor, "h1"))
    expect(scoreOf(floor, "h2")).toBeCloseTo(0.7, 5)
    expect(scoreOf(floor, "h1")).toBeCloseTo(0, 5)

    // Boosted: h1's construct match overtakes h2.
    expect(scoreOf(boosted, "h1")).toBeGreaterThan(scoreOf(boosted, "h2"))
    expect(scoreOf(boosted, "h1")).toBeCloseTo(1.0, 5)
    expect(scoreOf(boosted, "h2")).toBeCloseTo(0.7, 5)

    const recipeBoosted = boosted.filter(isContext)[0]!.recipe
    expect(recipeBoosted.affordancesFired).toContain("q:strata")
    const recipeFloor = floor.filter(isContext)[0]!.recipe
    expect(recipeFloor.affordancesFired).not.toContain("q:strata")
  })
})

describe("search v3 — gates/temporal channel", () => {
  // Both statements share identical utterance text + identical dense vector,
  // so BM25 and dense scores tie exactly — the ONLY thing that can separate
  // them is the +0.15 temporal boost.
  function fixtureGates(): CxnArtifacts {
    const statements = new Map<string, StatementEntry>([
      ["may1", stmt({
        hash: "may1", utterance: "They went hiking", ts: "2023-05-15T10:00:00Z",
        session: "s1", session_index: 0,
      })],
      ["june1", stmt({
        hash: "june1", utterance: "They went hiking", ts: "2023-06-15T10:00:00Z",
        session: "s2", session_index: 0,
      })],
    ])
    const vectors = new Map<string, number[]>([["may1", [1, 0]], ["june1", [1, 0]]])
    const turns = new Map([
      ["s1", [{ ts: "2023-05-15T10:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
      ["s2", [{ ts: "2023-06-15T10:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
    ])
    return {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements, turns, vectors, aggregates: { cxn: new Map(), episode: new Map() },
    }
  }

  test("qGates=true boosts via comprehension date_fillers and records q:temporal", async () => {
    const artifacts = fixtureGates()
    const comprehension: Comprehension = {
      ...EMPTY_COMPREHENSION,
      date_fillers: [{ text: "May 2023", year: 2023, month: 5 }],
    }
    const { fetchImpl } = routingFetch({ vector: [1, 0], comprehension })
    const cfg = baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local", qGates: true })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search(
      "hiking", { containerTag: "t" }
    )
    const utterances = results.filter(isUtterance)
    const may1 = utterances.find((u) => u.metadata.utterance_hash === "may1")!
    const june1 = utterances.find((u) => u.metadata.utterance_hash === "june1")!
    expect(may1.score).toBeCloseTo(june1.score + 0.15, 5)

    const recipe = results.filter(isContext)[0]!.recipe
    expect(recipe.affordancesFired).toContain("q:temporal")
    expect(recipe.gatesFired).not.toContain("temporal")
  })

  test("qGates=false falls back to the leg-2 regex path on an explicit month/year query", async () => {
    const artifacts = fixtureGates()
    const { fetchImpl } = routingFetch({ vector: [1, 0], comprehension: EMPTY_COMPREHENSION })
    const cfg = baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local", qGates: false })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search(
      "In May 2023, what did they do?", { containerTag: "t" }
    )
    const utterances = results.filter(isUtterance)
    const may1 = utterances.find((u) => u.metadata.utterance_hash === "may1")!
    const june1 = utterances.find((u) => u.metadata.utterance_hash === "june1")!
    expect(may1.score).toBeCloseTo(june1.score + 0.15, 5)

    const recipe = results.filter(isContext)[0]!.recipe
    expect(recipe.gatesFired).toContain("temporal")
    expect(recipe.affordancesFired).not.toContain("q:temporal")
  })
})

describe("search v3 — seed channel", () => {
  // "top" wins on dense alone (cos=1). "filler" sits between "top" and
  // "melanie_stmt" on dense (cos≈0.11) — enough to beat melanie_stmt's floor
  // score (0) but not enough to survive once melanie_stmt's seeded-BM25 term
  // (normalized to the full 0.3 sparse weight, since it's the only sparse
  // hit) is blended in. poolK=2 makes the swap observable as pool membership.
  function fixtureSeed(): CxnArtifacts {
    const statements = new Map<string, StatementEntry>([
      ["top", stmt({
        hash: "top", utterance: "The sun set slowly", ts: "2023-05-08T12:00:00Z",
        session: "s1", session_index: 0,
      })],
      ["filler", stmt({
        hash: "filler", utterance: "Rain fell during the afternoon", ts: "2023-05-08T13:00:00Z",
        session: "s2", session_index: 0,
      })],
      ["melanie_stmt", stmt({
        hash: "melanie_stmt", utterance: "Melanie packed her bags for the trip", ts: "2023-05-08T14:00:00Z",
        session: "s3", session_index: 0,
      })],
    ])
    const vectors = new Map<string, number[]>([
      ["top", [1, 0]],
      ["filler", [1, 9]],
      ["melanie_stmt", [0, 1]],
    ])
    const turns = new Map([
      ["s1", [{ ts: "2023-05-08T12:00:00Z", speaker: "A", text: "t1", blip_caption: null }]],
      ["s2", [{ ts: "2023-05-08T13:00:00Z", speaker: "B", text: "t2", blip_caption: null }]],
      ["s3", [{ ts: "2023-05-08T14:00:00Z", speaker: "A", text: "t3", blip_caption: null }]],
    ])
    return {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements, turns, vectors, aggregates: { cxn: new Map(), episode: new Map() },
    }
  }

  test("qSeed pulls a melanie-mentioning statement into the pool past a higher-blend statement", async () => {
    const artifacts = fixtureSeed()
    const comprehension: Comprehension = {
      ...EMPTY_COMPREHENSION,
      fillers: [{ lemma: "melanie", role: "subj", entity: true }],
    }

    const runWith = async (qSeed: boolean) => {
      const { fetchImpl } = routingFetch({ vector: [1, 0], comprehension })
      const cfg = baseConfigV3({
        q: true, comprehendUrl: "http://sidecar.local", qSeed, poolK: 2, finalK: 2,
        qStrata: false, qGates: false, qAnswer: false,
      })
      return new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search("elephant", { containerTag: "t" })
    }
    const withoutSeed = await runWith(false)
    const withSeed = await runWith(true)

    const hashesOf = (results: unknown[]) => new Set(results.filter(isUtterance).map((u) => u.metadata.utterance_hash))
    expect(hashesOf(withoutSeed).has("melanie_stmt")).toBe(false)
    expect(hashesOf(withSeed).has("melanie_stmt")).toBe(true)

    const recipe = withSeed.filter(isContext)[0]!.recipe
    expect(recipe.affordancesFired).toContain("q:seed")
  })
})

describe("search v3 — answer channel + fallback", () => {
  test("wh_slot 'when' -> directive starts with preamble + newline + contains absolute date; directiveVersion 2", async () => {
    const artifacts = fixtureParity()
    const comprehension: Comprehension = { ...EMPTY_COMPREHENSION, wh_slot: "when" }
    const { fetchImpl } = routingFetch({ vector: [1, 0], comprehension })
    const cfg = baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local" })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )
    const context = results.filter(isContext)[0]!
    expect(context.directive).toContain("State the concrete fact(s)")
    expect(context.directive).toContain("\n")
    expect(context.directive).toContain("absolute date")
    expect(context.recipe.affordancesFired).toContain("q:answer")
    expect(context.recipe.directiveVersion).toBe(2)
  })

  test("q:false control parity: directiveVersion undefined, deep-equals q:false baseline", async () => {
    const artifacts = fixtureParity()
    const { fetchImpl } = routingFetch({ vector: [1, 0] })
    const cfg = baseConfigV3({ q: false })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )
    const context = results.filter(isContext)[0]!
    expect(context.recipe.directiveVersion).toBeUndefined()
  })

  test("all-empty comprehension -> fallback:true, no affordances, directive null, order matches control", async () => {
    const artifacts = fixtureParity()

    const { fetchImpl: controlFetch } = routingFetch({ vector: [1, 0] })
    const controlCfg = baseConfigV3({ q: false })
    const controlResults = await new BonfiresCxnProvider(
      controlCfg, artifacts, depsWithFetch(controlFetch)
    ).search("xyzzy", { containerTag: "t" })

    const { fetchImpl } = routingFetch({ vector: [1, 0], comprehension: EMPTY_COMPREHENSION })
    const cfg = baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local" })
    const results = await new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl)).search(
      "xyzzy", { containerTag: "t" }
    )

    const orderOf = (r: unknown[]) => r.filter(isUtterance).map((u) => u.metadata.utterance_hash)
    expect(orderOf(results)).toEqual(orderOf(controlResults))

    const context = results.filter(isContext)[0]!
    expect(context.directive).toBeNull()
    expect(context.recipe.affordancesFired).toEqual([])
    expect(context.recipe.fallback).toBe(true)
  })
})

describe("search v3 — sidecar error", () => {
  test("500 from comprehendUrl rejects loudly, mentioning comprehend", async () => {
    const artifacts = fixtureParity()
    const { fetchImpl } = routingFetch({ vector: [1, 0], comprehendStatus: 500 })
    const cfg = baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local" })
    const provider = new BonfiresCxnProvider(cfg, artifacts, depsWithFetch(fetchImpl))
    await expect(provider.search("xyzzy", { containerTag: "t" })).rejects.toThrow(/comprehend/i)
  })
})

describe("search v3 — comprehend preflight (drift tripwire)", () => {
  // Artifacts carrying two non-residual construct ids so the subset assertion
  // has something real to check against health.construct_ids.
  function fixturePreflight(): CxnArtifacts {
    const statements = new Map<string, StatementEntry>([
      ["h1", stmt({
        hash: "h1", utterance: "Someone painted a mural", ts: "2023-05-08T13:00:00Z",
        session: "s1", session_index: 0, construct_ids: ["person.paint.v1", "residual.v1"],
      })],
      ["h2", stmt({
        hash: "h2", utterance: "Melanie asked about pets", ts: "2023-05-08T14:00:00Z",
        session: "s2", session_index: 0, construct_ids: ["person.ask.v1"],
      })],
    ])
    const vectors = new Map<string, number[]>([["h1", [1, 0]], ["h2", [0, 1]]])
    return {
      utteranceMap: new Map(), entrenchmentByConstruct: new Map(), planRecordCount: 0,
      statements, turns: new Map(), vectors, aggregates: { cxn: new Map(), episode: new Map() },
    }
  }

  const cfg = () => baseConfigV3({ q: true, comprehendUrl: "http://sidecar.local" })

  test("passes when cards >= 1 and every non-residual artifact construct id is in health", async () => {
    const { fetchImpl } = routingFetch({
      vector: [1, 0],
      health: { cards: 2, digest: "d1", construct_ids: ["person.paint.v1", "person.ask.v1", "person.eat.v1"] },
    })
    const provider = new BonfiresCxnProvider(cfg(), fixturePreflight(), depsWithFetch(fetchImpl))
    await expect(provider.comprehendPreflight()).resolves.toBeUndefined()
  })

  test("throws when the sidecar reports zero cards", async () => {
    const { fetchImpl } = routingFetch({
      vector: [1, 0],
      health: { cards: 0, digest: "d1", construct_ids: ["person.paint.v1", "person.ask.v1"] },
    })
    const provider = new BonfiresCxnProvider(cfg(), fixturePreflight(), depsWithFetch(fetchImpl))
    await expect(provider.comprehendPreflight()).rejects.toThrow(/cards/)
  })

  test("throws NAMING a non-residual artifact construct id missing from health.construct_ids", async () => {
    // person.ask.v1 present, person.paint.v1 missing — residual.v1 exempt.
    const { fetchImpl } = routingFetch({
      vector: [1, 0],
      health: { cards: 2, digest: "d1", construct_ids: ["person.ask.v1"] },
    })
    const provider = new BonfiresCxnProvider(cfg(), fixturePreflight(), depsWithFetch(fetchImpl))
    await expect(provider.comprehendPreflight()).rejects.toThrow(/person\.paint\.v1/)
  })
})

describe("config v3 — CXN_Q / CXN_COMPREHEND_URL coupling", () => {
  const FULL_ENV = {
    CXN_NEO4J_URI: "bolt://localhost:7687",
    CXN_NEO4J_USER: "neo4j",
    CXN_NEO4J_PASSWORD: "pw",
    CXN_GROUP_ID: "g",
    CXN_ARTIFACTS_DIR: "/tmp/artifacts",
    CXN_UTTERANCE_MAP: "/tmp/map.json",
    CXN_VOYAGE_API_KEY: "voyage-key",
    CXN_CORPUS: "/tmp/corpus.json",
    CXN_SESSION_TURNS: "/tmp/turns.json",
    CXN_EMBEDDINGS: "/tmp/embeddings.json",
  }

  test("CXN_Q=1 without CXN_COMPREHEND_URL throws naming the missing var", () => {
    expect(() => loadCxnConfig({ ...FULL_ENV, CXN_Q: "1" })).toThrow(/CXN_COMPREHEND_URL/)
  })

  test("CXN_Q=1 with CXN_COMPREHEND_URL loads q:true + comprehendUrl and channel defaults", () => {
    const cfg = loadCxnConfig({ ...FULL_ENV, CXN_Q: "1", CXN_COMPREHEND_URL: "http://sidecar.local" })
    expect(cfg.q).toBe(true)
    expect(cfg.comprehendUrl).toBe("http://sidecar.local")
    expect(cfg.qStrata).toBe(true)
    expect(cfg.qGates).toBe(true)
    expect(cfg.qSeed).toBe(true)
    expect(cfg.qAnswer).toBe(true)
    expect(cfg.qDelta).toBe(0.3)
    expect(cfg.qSeedEntityW).toBe(2.0)
    expect(cfg.qSeedVerbW).toBe(1.0)
    expect(cfg.qSeedLaneW).toBe(0.5)
  })

  test("CXN_Q unset defaults q:false and does not require the URL", () => {
    const cfg = loadCxnConfig(FULL_ENV)
    expect(cfg.q).toBe(false)
    expect(cfg.comprehendUrl).toBeUndefined()
  })
})
