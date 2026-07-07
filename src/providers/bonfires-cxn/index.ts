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
import { FIRING_STRUCTURES, NEIGHBORS_TO_FIRINGS, SEED_ENTITIES, SEEDS_TO_FIRINGS } from "./cypher"
import {
  assembleResults,
  buildCxnAnswerPrompt,
  buildStructureItem,
  extractTerms,
  rankFirings,
  type CandidateFiring,
  type StructureLine,
} from "./retrieval"

export interface CxnDeps {
  runCypher: (query: string, params: Record<string, unknown>) => Promise<Record<string, unknown>[]>
}

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
  prompts = { answerPrompt: buildCxnAnswerPrompt }

  private cfg: CxnConfig | null
  private artifacts: CxnArtifacts | null
  private deps: CxnDeps | null
  private driver: Driver | null = null

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
    logger.info(`bonfires-cxn: preflight OK for group ${this.cfg.groupId}`)
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

  async search(query: string, _options: SearchOptions): Promise<unknown[]> {
    const { cfg, artifacts, deps } = this.requireState()
    const terms = extractTerms(query)
    if (terms.length === 0) return []

    const seeds = await deps.runCypher(SEED_ENTITIES, {
      groupId: cfg.groupId,
      terms,
      maxSeeds: neo4j.int(cfg.maxSeedEntities),
    })
    const seedUuids = seeds.map((row) => String(row.uuid))
    if (seedUuids.length === 0) return []

    const candidateByUuid = new Map<string, CandidateFiring>()
    for (const template of [SEEDS_TO_FIRINGS, NEIGHBORS_TO_FIRINGS]) {
      const rows = await deps.runCypher(template, { groupId: cfg.groupId, seedUuids })
      for (const row of rows) {
        const attrs = JSON.parse(String(row.attributes ?? "{}")) as {
          utterance_hash?: string
          ts?: string
        }
        if (!attrs.utterance_hash) continue
        const uuid = String(row.uuid)
        const matched = (row.matchedSeeds as unknown[]).map(String)
        const existing = candidateByUuid.get(uuid)
        if (existing) {
          for (const seed of matched) existing.matchedSeeds.add(seed)
        } else {
          candidateByUuid.set(uuid, {
            uuid,
            constructId: String(row.name).split("@")[0] ?? "",
            utteranceHash: attrs.utterance_hash,
            ts: attrs.ts ?? "",
            matchedSeeds: new Set(matched),
          })
        }
      }
    }

    const ranked = rankFirings([...candidateByUuid.values()], artifacts.entrenchmentByConstruct, cfg.topKFirings)
    if (ranked.length === 0) return []

    const structureRows = await deps.runCypher(FIRING_STRUCTURES, {
      groupId: cfg.groupId,
      firingUuids: ranked.map((f) => f.uuid),
    })
    const structuresByEvent = new Map<string, StructureLine>()
    for (const row of structureRows) {
      const key = String(row.eventUuid)
      const attrs = JSON.parse(String(row.eventAttributes ?? "{}")) as { predicate?: string; ts?: string }
      const firing = candidateByUuid.get(String(row.firingUuid))
      const line = structuresByEvent.get(key) ?? {
        firingUuid: String(row.firingUuid),
        predicate: attrs.predicate ?? "",
        roles: [],
        ts: firing?.ts ?? "",
        constructId: firing?.constructId ?? "",
      }
      line.roles.push({ role: String(row.role), filler: String(row.filler) })
      structuresByEvent.set(key, line)
    }

    const utteranceResults = assembleResults(ranked, artifacts.utteranceMap, [])
    const structureItem = buildStructureItem([...structuresByEvent.values()])
    return [...utteranceResults, structureItem]
  }

  async clear(containerTag: string): Promise<void> {
    // NEVER delete the KG under test — the fold is the system under test, not run state.
    logger.warn(`bonfires-cxn: clear(${containerTag}) refused — gate-B KG is read-only for this provider`)
  }

  protected requireState(): { cfg: CxnConfig; artifacts: CxnArtifacts; deps: CxnDeps } {
    if (!this.cfg || !this.artifacts || !this.deps) throw new Error("bonfires-cxn: provider not initialized")
    return { cfg: this.cfg, artifacts: this.artifacts, deps: this.deps }
  }
}

export default BonfiresCxnProvider
