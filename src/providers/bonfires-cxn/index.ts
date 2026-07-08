import neo4j, { type Driver } from "neo4j-driver"
import type {
  IndexingProgressCallback,
  IngestOptions,
  IngestResult,
  Provider,
  ProviderConfig,
  SearchOptions,
} from "../../types/provider"
import type { UnifiedSession } from "../../types/unified"
import { logger } from "../../utils/logger"
import { loadCxnConfig, type CxnConfig } from "./config"
import { loadArtifacts, type CxnArtifacts } from "./artifacts"
import { embedTexts, VOYAGE_MODEL, type FetchLike } from "./voyage"
import {
  applyTemporalBoost,
  blendScores,
  bm25Scores,
  bm25ScoresWeighted,
  buildAnswerPromptV3,
  buildBm25,
  CORPUS_YEAR,
  denseScores,
  hydrationLines,
  lanePBoost,
  mmrSelect,
  replyExpansion,
  temporalWindow,
  type Aggregates,
  type Bm25Index,
  type StatementEntry,
  type Turn,
} from "./retrieval2"
import {
  answerDirective,
  directivePreamble,
  combineSparse,
  isFallback,
  seededTermWeights,
  strataBoost,
  temporalWindowFromDates,
  type AffordanceKey,
  type Comprehension,
} from "./affordance"

export interface CxnDeps {
  runCypher: (query: string, params: Record<string, unknown>) => Promise<Record<string, unknown>[]>
  fetchImpl?: FetchLike
}

type CxnConfigV2 = CxnConfig &
  Required<
    Pick<
      CxnConfig,
      | "voyageApiKey"
      | "corpusPath"
      | "sessionTurnsPath"
      | "embeddingsPath"
      | "laneP"
      | "blendDense"
      | "blendSparse"
      | "poolK"
      | "finalK"
      | "deltaCxn"
      | "deltaEp"
      | "hydrateTop"
      | "hydrateWindow"
      | "q"
      | "qStrata"
      | "qGates"
      | "qSeed"
      | "qAnswer"
      | "qDelta"
      | "qSeedEntityW"
      | "qSeedVerbW"
      | "qSeedLaneW"
      | "mmr"
      | "mmrLambda"
      | "captions"
      | "captionDamp"
    >
  >

interface SidecarHealth {
  cards: number
  digest: string
  construct_ids: string[]
}

type CxnArtifactsV2 = CxnArtifacts &
  Required<Pick<CxnArtifacts, "statements" | "turns" | "vectors" | "aggregates">>

const COUNT_NODES = `MATCH (n:Entity {group_id: $groupId}) RETURN count(n) AS n`
const COUNT_EDGES = `MATCH (a:Entity {group_id: $groupId})-[r]->(b:Entity {group_id: $groupId}) RETURN count(r) AS n`
const COUNT_FIRINGS = `MATCH (f:Entity_Firing {group_id: $groupId}) RETURN count(f) AS n`
const SAMPLE_FIRINGS = `MATCH (f:Entity_Firing {group_id: $groupId})
RETURN f.uuid AS uuid, f.attributes AS attributes ORDER BY f.uuid ASC LIMIT 10`

function driverDeps(driver: Driver): CxnDeps {
  return {
    runCypher: async (query, params) => {
      const session = driver.session()
      try {
        const result = await session.run(query, params)
        return result.records.map((record) => {
          const row: Record<string, unknown> = {}
          for (const key of record.keys) {
            const value = record.get(key)
            row[String(key)] = neo4j.isInt(value) ? value.toNumber() : value
          }
          return row
        })
      } finally {
        await session.close()
      }
    },
  }
}

export class BonfiresCxnProvider implements Provider {
  name = "bonfires-cxn"
  concurrency = { default: 5, ingest: 1 }
  prompts = { answerPrompt: buildAnswerPromptV3 }

  private cfg: CxnConfig | null
  private artifacts: CxnArtifacts | null
  private deps: CxnDeps | null
  private driver: Driver | null = null
  private bm25: Bm25Index | null = null
  private captionBm25: Bm25Index | null = null
  private queryVectorMemo = new Map<string, number[]>()
  private comprehensionMemo = new Map<string, Comprehension>()

  // Test constructor: inject everything. Production path: no-arg + initialize().
  constructor(cfg?: CxnConfig, artifacts?: CxnArtifacts, deps?: CxnDeps) {
    this.cfg = cfg ?? null
    this.artifacts = artifacts ?? null
    this.deps = deps ?? null
  }

  async initialize(_config: ProviderConfig): Promise<void> {
    if (!this.cfg) this.cfg = loadCxnConfig()
    if (!this.artifacts) this.artifacts = await loadArtifacts(this.cfg)
    if (!this.deps) {
      this.driver = neo4j.driver(
        this.cfg.neo4jUri,
        neo4j.auth.basic(this.cfg.neo4jUser, this.cfg.neo4jPassword)
      )
      this.deps = driverDeps(this.driver)
    }
    await this.preflight()

    const { cfg } = this.requireState2()
    this.ensureBm25()
    this.ensureCaptionBm25()

    // Probe the Voyage key once, hard fail if it doesn't work.
    await embedTexts(["probe"], "query", cfg.voyageApiKey, this.deps.fetchImpl ?? (globalThis.fetch as FetchLike))

    if (cfg.q) await this.comprehendPreflight()

    logger.info(`bonfires-cxn: preflight OK for group ${cfg.groupId}`)
  }

  // Sidecar drift tripwire: every non-residual.v1 construct id that actually
  // appears in the fold artifacts must be known to the live comprehend
  // sidecar's grammar (health.construct_ids), or comprehension results would
  // silently never strata-match against them. Public for the same reason
  // preflight() is: testable with injected cfg/artifacts/deps, no driver.
  async comprehendPreflight(): Promise<void> {
    const { cfg, artifacts, deps } = this.requireState2()
    if (!cfg.comprehendUrl) throw new Error("bonfires-cxn: CXN_Q=1 requires comprehendUrl")
    const fetchImpl = deps.fetchImpl ?? (globalThis.fetch as FetchLike)
    const response = await fetchImpl(`${cfg.comprehendUrl}/health`, { method: "GET" })
    if (!response.ok) {
      throw new Error(`bonfires-cxn: comprehend sidecar /health failed (${response.status})`)
    }
    const health = (await response.json()) as SidecarHealth
    if (!(health.cards >= 1)) {
      throw new Error(`bonfires-cxn: comprehend sidecar reports ${health.cards} cards — grammar not loaded`)
    }
    const healthIds = new Set(health.construct_ids)
    const missing = new Set<string>()
    for (const entry of artifacts.statements.values()) {
      for (const id of entry.construct_ids) {
        if (id !== "residual.v1" && !healthIds.has(id)) missing.add(id)
      }
    }
    if (missing.size > 0) {
      throw new Error(
        `bonfires-cxn: comprehend sidecar missing construct ids present in artifacts: ${[...missing].sort().join(", ")}`
      )
    }
  }

  // Built once (lazily). initialize() calls this eagerly for the production
  // path; the test constructor path (search2.test.ts) builds it on first
  // search() so tests can exercise search() directly without a Cypher-backed
  // initialize()/preflight() round trip.
  private ensureBm25(): Bm25Index {
    if (this.bm25) return this.bm25
    const { artifacts } = this.requireState2()
    this.bm25 = buildBm25([...artifacts.statements.values()])
    return this.bm25
  }

  // Caption lane's own BM25 index — a SEPARATE index from ensureBm25()'s
  // statement index (never merged), which is what makes the caption-isolation
  // invariant (statement BM25 scores identical whether cfg.captions is on or
  // off) hold structurally rather than by coincidence. Reuses buildBm25 via
  // minimal StatementEntry-shaped literals: BM25 only reads hash+utterance,
  // the rest (ts/actor_id/session) just satisfies the type and threads
  // through unused. Returns null when captions are off or the artifact wasn't
  // loaded, mirroring the two-flag guard search() itself uses.
  private ensureCaptionBm25(): Bm25Index | null {
    const { cfg, artifacts } = this.requireState2()
    if (!cfg.captions || !artifacts.captions) return null
    if (this.captionBm25) return this.captionBm25
    this.captionBm25 = buildBm25(
      [...artifacts.captions].map(
        ([id, c]): StatementEntry => ({
          hash: id,
          utterance: `[image] ${c.caption}`,
          ts: c.ts,
          actor_id: c.actor_id,
          session: c.session,
          session_index: 0,
          construct_ids: [],
        })
      )
    )
    return this.captionBm25
  }

  async preflight(): Promise<void> {
    const { cfg, artifacts, deps } = this.requireState()
    const [nodes] = await deps.runCypher(COUNT_NODES, { groupId: cfg.groupId })
    if (Number(nodes?.n) !== cfg.expectedNodes) {
      throw new Error(
        `bonfires-cxn preflight: group ${cfg.groupId} has ${String(nodes?.n)} nodes, expectedNodes=${cfg.expectedNodes}`
      )
    }
    const [edges] = await deps.runCypher(COUNT_EDGES, { groupId: cfg.groupId })
    if (Number(edges?.n) !== cfg.expectedEdges) {
      throw new Error(
        `bonfires-cxn preflight: group ${cfg.groupId} has ${String(edges?.n)} edges, expectedEdges=${cfg.expectedEdges}`
      )
    }
    const [firings] = await deps.runCypher(COUNT_FIRINGS, { groupId: cfg.groupId })
    if (Number(firings?.n) !== artifacts.planRecordCount) {
      throw new Error(
        `bonfires-cxn preflight: ${String(firings?.n)} firings vs ${artifacts.planRecordCount} plan records — artifact drift`
      )
    }
    const sampled = await deps.runCypher(SAMPLE_FIRINGS, { groupId: cfg.groupId })
    for (const row of sampled) {
      const attrs = JSON.parse(String(row.attributes ?? "{}")) as { utterance_hash?: string }
      if (!attrs.utterance_hash || !artifacts.utteranceMap.has(attrs.utterance_hash)) {
        throw new Error(
          `bonfires-cxn preflight: firing ${String(row.uuid)} hash ${String(attrs.utterance_hash)} missing from sidecar map`
        )
      }
    }
  }

  async ingest(_sessions: UnifiedSession[], _options: IngestOptions): Promise<IngestResult> {
    // KG is pre-built (gate-B fold) and integrity-checked in initialize().
    return { documentIds: [] }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    onProgress?.({ completedIds: result.documentIds, failedIds: [], total: result.documentIds.length })
  }

  private async embedQuery(query: string): Promise<number[]> {
    const key = new Bun.CryptoHasher("sha256").update(query).digest("hex")
    const cached = this.queryVectorMemo.get(key)
    if (cached) return cached
    const { cfg, artifacts, deps } = this.requireState2()
    const fetchImpl = deps.fetchImpl ?? (globalThis.fetch as FetchLike)
    const [vector] = await embedTexts([query], "query", cfg.voyageApiKey, fetchImpl)
    if (!vector) throw new Error("bonfires-cxn: embedQuery got no vector back from voyage")
    // Hard-fail a live query vector whose dimension doesn't match the corpus
    // embeddings' own dim — a silent mismatch would rank everything via NaN
    // cosine similarity instead of erroring. artifacts.dim is only absent for
    // bare-literal test fixtures that bypass loadArtifacts(); the guard is a
    // no-op for those, matching their existing (already-consistent) vectors.
    if (artifacts.dim !== undefined && vector.length !== artifacts.dim) {
      throw new Error(
        `bonfires-cxn: embedQuery vector dimension mismatch — got length ${vector.length}, expected ${artifacts.dim}`
      )
    }
    this.queryVectorMemo.set(key, vector)
    return vector
  }

  private async comprehendQuery(query: string): Promise<Comprehension> {
    const cached = this.comprehensionMemo.get(query)
    if (cached) return cached
    const { cfg, deps } = this.requireState2()
    if (!cfg.comprehendUrl) throw new Error("bonfires-cxn: CXN_Q=1 requires comprehendUrl")
    const fetchImpl = deps.fetchImpl ?? (globalThis.fetch as FetchLike)
    const response = await fetchImpl(`${cfg.comprehendUrl}/comprehend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query }),
    })
    if (!response.ok) {
      throw new Error(`bonfires-cxn: comprehend sidecar failed (${response.status}) — refusing silent floor fallback`)
    }
    const comprehension = (await response.json()) as Comprehension
    this.comprehensionMemo.set(query, comprehension)
    return comprehension
  }

  async search(query: string, _options: SearchOptions): Promise<unknown[]> {
    const { cfg, artifacts } = this.requireState2()
    const [queryVector, comprehension] = await Promise.all([
      this.embedQuery(query), // live, memoized, HARD fail
      cfg.q ? this.comprehendQuery(query) : Promise.resolve(null),
    ])
    const affordancesFired: AffordanceKey[] = []

    // ---- seed channel (before blend) ----
    let sparse = bm25Scores(this.ensureBm25(), query)
    if (cfg.q && cfg.qSeed && comprehension && comprehension.fillers.length) {
      const seeded = bm25ScoresWeighted(
        this.ensureBm25(),
        seededTermWeights(comprehension.fillers, cfg.qSeedEntityW, cfg.qSeedVerbW)
      )
      sparse = combineSparse(sparse, seeded, cfg.qSeedLaneW)
      affordancesFired.push("q:seed")
    }

    // ---- blend + lane P (unchanged) ----
    const dense = denseScores(queryVector, artifacts.vectors)
    let scores = blendScores(dense, sparse, cfg.blendDense, cfg.blendSparse)
    const gatesFired: string[] = []
    let lanePTop: Array<{ key: string; sim: number }> = []
    if (cfg.laneP) {
      const { boosted, topContribs } = lanePBoost(
        scores,
        queryVector,
        artifacts.aggregates,
        artifacts.statements,
        cfg.deltaCxn,
        cfg.deltaEp
      )
      scores = boosted
      lanePTop = topContribs
      gatesFired.push("laneP")
    }

    // ---- strata channel (after blend, before pool) ----
    if (cfg.q && cfg.qStrata && comprehension && comprehension.matched_cxn_ids.length) {
      scores = strataBoost(scores, artifacts.statements, comprehension.matched_cxn_ids, cfg.qDelta)
      affordancesFired.push("q:strata")
    }

    // ---- caption lane (leg 5, spec §0.4): candidates merged in AFTER strata,
    // BEFORE temporal — captions never receive the strata/seed boosts above
    // (those loops key off statement construct_ids/fillers, and caption ids
    // don't exist in `scores` yet when those channels ran), but DO compete
    // for the temporal boost, pool cut, and final-K selection below like any
    // other candidate. A separate dense/sparse/blend pass over the caption
    // artifact (own BM25 index, own vectors) keeps this a structurally
    // separate-index lane rather than a merge into the statement lane — the
    // caption-isolation invariant (statement BM25 unchanged by cfg.captions)
    // follows from never touching `this.bm25`/`sparse`/`dense` here.
    let captionLanes: { dense: Map<string, number>; sparse: Map<string, number> } | null = null
    let captionsInPool = 0
    if (cfg.captions && artifacts.captions && artifacts.captionVectors) {
      const captionBm25 = this.ensureCaptionBm25()!
      const captionDense = denseScores(queryVector, artifacts.captionVectors)
      const captionSparse = bm25Scores(captionBm25, query)
      const captionBlended = blendScores(captionDense, captionSparse, cfg.blendDense, cfg.blendSparse)
      for (const [id, score] of captionBlended) scores.set(id, score * cfg.captionDamp)
      captionLanes = { dense: captionDense, sparse: captionSparse }
    }

    // ---- temporal: comprehension-derived window when q+qGates, else legacy
    // regex. Unified id->ts map (statements ∪ captions) so a caption can be
    // temporally boosted the same as a statement — otherwise a caption whose
    // ts falls in the query's date window would silently never be reachable
    // via that channel.
    const tsById: Map<string, { ts: string }> = new Map(artifacts.statements)
    if (cfg.captions && artifacts.captions) {
      for (const [id, c] of artifacts.captions) tsById.set(id, { ts: c.ts })
    }
    if (cfg.q && cfg.qGates) {
      // Deliberate spec deviation from §0.3(2): the gate fires only when
      // date_fillers is nonempty. A bare wh_slot === "when" derives no window
      // on its own (there's no date text to build one from) — the spec
      // wording predates this comprehension shape.
      const window = comprehension ? temporalWindowFromDates(comprehension.date_fillers, CORPUS_YEAR) : null
      if (window) {
        scores = applyTemporalBoost(scores, tsById, window, 0.15)
        affordancesFired.push("q:temporal")
      }
    } else {
      const window = temporalWindow(query)
      if (window) {
        scores = applyTemporalBoost(scores, tsById, window, 0.15)
        gatesFired.push("temporal")
      }
    }

    const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    const pool = ranked.slice(0, cfg.poolK).map(([hash]) => hash)
    const poolScores = new Map(ranked.slice(0, cfg.poolK))
    const { added } = replyExpansion(pool, poolScores, artifacts.statements, 10, 0.8)
    if (added.length) gatesFired.push("reply")
    for (const { hash, score } of added) poolScores.set(hash, score)
    if (cfg.captions && artifacts.captions) {
      for (const hash of poolScores.keys()) if (artifacts.captions.has(hash)) captionsInPool += 1
    }

    // ---- final-K selection: MMR diversification (list-shape questions only)
    // or the existing score-sort. mmrSelect only reorders WITHIN the pool —
    // never imports new candidates — so `final ids ⊆ pool` holds by
    // construction either way.
    const mmrFired = cfg.mmr && cfg.q && comprehension?.wh_slot === "list"
    const final = mmrFired
      ? mmrSelect(
          poolScores,
          (id) => artifacts.vectors.get(id) ?? artifacts.captionVectors?.get(id),
          cfg.finalK,
          cfg.mmrLambda
        )
      : [...poolScores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, cfg.finalK)
    if (mmrFired) affordancesFired.push("b:mmr")
    const finalHashes = final.map(([hash]) => hash)
    // hydrationLines keys off artifacts.statements only — caption ids skip
    // naturally (artifacts.statements.get(captionHash) misses -> `continue`).
    const lines = hydrationLines(finalHashes.slice(0, cfg.hydrateTop), artifacts.statements, artifacts.turns, cfg.hydrateWindow)

    // ---- answer directive ----
    const slotDirective = cfg.q && cfg.qAnswer ? answerDirective(comprehension?.wh_slot ?? null) : null
    const directive = slotDirective ? `${directivePreamble()}\n${slotDirective}` : null
    if (directive) affordancesFired.push("q:answer")

    const querySha = new Bun.CryptoHasher("sha256").update(JSON.stringify(queryVector)).digest("hex").slice(0, 16)
    const tsOf = (hash: string) => artifacts.statements.get(hash)?.ts ?? artifacts.captions?.get(hash)?.ts ?? ""
    const utteranceItems = final
      .sort((a, b) => {
        const ta = tsOf(a[0]), tb = tsOf(b[0])
        return ta < tb ? -1 : ta > tb ? 1 : a[0] < b[0] ? -1 : 1
      })
      .map(([hash, score]) => {
        const entry = artifacts.statements.get(hash)
        if (entry) {
          return {
            text: `[${entry.ts.slice(0, 16).replace("T", " ")} ${entry.actor_id}] ${entry.utterance}`,
            kind: "cxn_utterance" as const,
            score,
            metadata: {
              utterance_hash: hash,
              construct_ids: entry.construct_ids,
              session: entry.session,
              lanes: { dense: dense.get(hash) ?? 0, sparse: sparse.get(hash) ?? 0 },
              query_vector_sha256: querySha,
            },
          }
        }
        // Caption item — artifacts.statements missed, so this hash must be a
        // caption id (the only other candidate source competing in `scores`).
        const caption = artifacts.captions!.get(hash)!
        return {
          text: `[${caption.ts.slice(0, 16).replace("T", " ")} ${caption.actor_id}] (image) ${caption.caption}`,
          kind: "cxn_utterance" as const,
          score,
          metadata: {
            caption_id: hash,
            lanes: { dense: captionLanes?.dense.get(hash) ?? 0, sparse: captionLanes?.sparse.get(hash) ?? 0 },
          },
        }
      })
    const inFinalK = artifacts.captions ? finalHashes.filter((hash) => artifacts.captions!.has(hash)).length : 0
    if (inFinalK > 0) affordancesFired.push("b:captions")

    const baseRecipe = {
      model: VOYAGE_MODEL,
      laneP: cfg.laneP,
      blendDense: cfg.blendDense,
      blendSparse: cfg.blendSparse,
      poolK: cfg.poolK,
      finalK: cfg.finalK,
      deltaCxn: cfg.deltaCxn,
      deltaEp: cfg.deltaEp,
      gatesFired,
      lanePTop,
    }
    const recipe = cfg.q
      ? {
          ...baseRecipe,
          q: cfg.q,
          qStrata: cfg.qStrata,
          qGates: cfg.qGates,
          qSeed: cfg.qSeed,
          qAnswer: cfg.qAnswer,
          qDelta: cfg.qDelta,
          qSeedEntityW: cfg.qSeedEntityW,
          qSeedVerbW: cfg.qSeedVerbW,
          qSeedLaneW: cfg.qSeedLaneW,
          directiveVersion: 2,
          affordancesFired,
          fallback: comprehension ? isFallback(comprehension) : false,
          comprehend: comprehension
            ? { probe: comprehension.probe, matched_cxn_ids: comprehension.matched_cxn_ids, wh_slot: comprehension.wh_slot }
            : null,
          ...(cfg.mmr
            ? {
                mmr: {
                  enabled: cfg.mmr,
                  lambda: cfg.mmrLambda,
                  ...(mmrFired ? { pool: [...poolScores.keys()].sort() } : {}),
                },
              }
            : {}),
          ...(cfg.captions ? { captions: { loaded: artifacts.captions?.size ?? 0, inPool: captionsInPool, inFinalK } } : {}),
        }
      : baseRecipe

    return [
      ...utteranceItems,
      {
        kind: "cxn_context",
        lines,
        directive,
        recipe,
      },
    ]
  }

  async clear(containerTag: string): Promise<void> {
    // NEVER delete the KG under test — the fold is the system under test, not run state.
    logger.warn(`bonfires-cxn: clear(${containerTag}) refused — gate-B KG is read-only for this provider`)
  }

  protected requireState(): { cfg: CxnConfig; artifacts: CxnArtifacts; deps: CxnDeps } {
    if (!this.cfg || !this.artifacts || !this.deps) throw new Error("bonfires-cxn: provider not initialized")
    return { cfg: this.cfg, artifacts: this.artifacts, deps: this.deps }
  }

  protected requireState2(): { cfg: CxnConfigV2; artifacts: CxnArtifactsV2; deps: CxnDeps } {
    const { cfg, artifacts, deps } = this.requireState()
    if (
      cfg.voyageApiKey === undefined ||
      cfg.corpusPath === undefined ||
      cfg.sessionTurnsPath === undefined ||
      cfg.embeddingsPath === undefined ||
      cfg.laneP === undefined ||
      cfg.blendDense === undefined ||
      cfg.blendSparse === undefined ||
      cfg.poolK === undefined ||
      cfg.finalK === undefined ||
      cfg.deltaCxn === undefined ||
      cfg.deltaEp === undefined ||
      cfg.hydrateTop === undefined ||
      cfg.hydrateWindow === undefined
    ) {
      throw new Error("bonfires-cxn: provider not initialized with v2 config")
    }
    if (!artifacts.statements || !artifacts.turns || !artifacts.vectors || !artifacts.aggregates) {
      throw new Error("bonfires-cxn: provider not initialized with v2 artifacts")
    }
    // v3 fields are defaulted here (mirroring loadCxnConfig's env defaults)
    // rather than strictly required, so pre-v3 fixtures that build a bare
    // CxnConfig literal (search2.test.ts, preflight.test.ts, config.test.ts,
    // retrieval.test.ts) keep compiling AND behaving byte-identically —
    // control parity holds whether q is `undefined` or explicit `false`.
    const cfgV3: CxnConfigV2 = {
      ...cfg,
      q: cfg.q ?? false,
      qStrata: cfg.qStrata ?? true,
      qGates: cfg.qGates ?? true,
      qSeed: cfg.qSeed ?? true,
      qAnswer: cfg.qAnswer ?? true,
      qDelta: cfg.qDelta ?? 0.3,
      qSeedEntityW: cfg.qSeedEntityW ?? 2.0,
      qSeedVerbW: cfg.qSeedVerbW ?? 1.0,
      qSeedLaneW: cfg.qSeedLaneW ?? 0.5,
      // v4 (leg 5) — same `??` defaulting pattern as the v3 block above.
      mmr: cfg.mmr ?? false,
      mmrLambda: cfg.mmrLambda ?? 0.3,
      captions: cfg.captions ?? false,
      captionDamp: cfg.captionDamp ?? 1.0,
    } as CxnConfigV2
    if (cfgV3.q && !cfgV3.comprehendUrl) {
      throw new Error("bonfires-cxn: CXN_Q=1 requires comprehendUrl")
    }
    return { cfg: cfgV3, artifacts: artifacts as CxnArtifactsV2, deps }
  }
}

export default BonfiresCxnProvider
