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
  buildAnswerPromptV2,
  buildBm25,
  denseScores,
  hydrationLines,
  lanePBoost,
  replyExpansion,
  temporalWindow,
  type Aggregates,
  type Bm25Index,
  type StatementEntry,
  type Turn,
} from "./retrieval2"

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
    >
  >

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
  prompts = { answerPrompt: buildAnswerPromptV2 }

  private cfg: CxnConfig | null
  private artifacts: CxnArtifacts | null
  private deps: CxnDeps | null
  private driver: Driver | null = null
  private bm25: Bm25Index | null = null
  private queryVectorMemo = new Map<string, number[]>()

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

    // Probe the Voyage key once, hard fail if it doesn't work.
    await embedTexts(["probe"], "query", cfg.voyageApiKey, this.deps.fetchImpl ?? (globalThis.fetch as FetchLike))

    logger.info(`bonfires-cxn: preflight OK for group ${cfg.groupId}`)
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
    const { cfg, deps } = this.requireState2()
    const fetchImpl = deps.fetchImpl ?? (globalThis.fetch as FetchLike)
    const [vector] = await embedTexts([query], "query", cfg.voyageApiKey, fetchImpl)
    if (!vector) throw new Error("bonfires-cxn: embedQuery got no vector back from voyage")
    this.queryVectorMemo.set(key, vector)
    return vector
  }

  async search(query: string, _options: SearchOptions): Promise<unknown[]> {
    const { cfg, artifacts } = this.requireState2()
    const queryVector = await this.embedQuery(query) // live, memoized, HARD fail
    const sparse = bm25Scores(this.ensureBm25(), query)
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
    const window = temporalWindow(query)
    if (window) {
      scores = applyTemporalBoost(scores, artifacts.statements, window, 0.15)
      gatesFired.push("temporal")
    }
    const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    const pool = ranked.slice(0, cfg.poolK).map(([hash]) => hash)
    const poolScores = new Map(ranked.slice(0, cfg.poolK))
    const { added } = replyExpansion(pool, poolScores, artifacts.statements, 10, 0.8)
    if (added.length) gatesFired.push("reply")
    for (const { hash, score } of added) poolScores.set(hash, score)
    const final = [...poolScores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, cfg.finalK)
    const finalHashes = final.map(([hash]) => hash)
    const lines = hydrationLines(finalHashes.slice(0, cfg.hydrateTop), artifacts.statements, artifacts.turns, cfg.hydrateWindow)

    const querySha = new Bun.CryptoHasher("sha256").update(JSON.stringify(queryVector)).digest("hex").slice(0, 16)
    const utteranceItems = final
      .sort((a, b) => {
        const ta = artifacts.statements.get(a[0])?.ts ?? "",
          tb = artifacts.statements.get(b[0])?.ts ?? ""
        return ta < tb ? -1 : ta > tb ? 1 : a[0] < b[0] ? -1 : 1
      })
      .map(([hash, score]) => {
        const entry = artifacts.statements.get(hash)!
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
      })
    return [
      ...utteranceItems,
      {
        kind: "cxn_context",
        lines,
        recipe: {
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
        },
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
    return { cfg: cfg as CxnConfigV2, artifacts: artifacts as CxnArtifactsV2, deps }
  }
}

export default BonfiresCxnProvider
