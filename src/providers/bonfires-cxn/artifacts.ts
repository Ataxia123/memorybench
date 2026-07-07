import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { CxnConfig } from "./config"
import { VOYAGE_MODEL } from "./voyage"
import type { Aggregates, StatementEntry, Turn } from "./retrieval2"

export interface UtteranceEntry {
  utterance: string
  ts: string
  actor_id: string
  session: string
}

export interface CxnArtifacts {
  utteranceMap: Map<string, UtteranceEntry>
  entrenchmentByConstruct: Map<string, number>
  planRecordCount: number
  // v2 (live-query dense floor + lane P search). Optional so pre-existing
  // leg-1 fixtures (preflight.test.ts) that build a bare CxnArtifacts literal
  // keep compiling unchanged; loadArtifacts() always populates them, and
  // requireState2() asserts + narrows them at runtime.
  statements?: Map<string, StatementEntry>
  turns?: Map<string, Turn[]>
  vectors?: Map<string, number[]>
  aggregates?: Aggregates
}

export async function loadArtifacts(cfg: CxnConfig): Promise<CxnArtifacts> {
  const mapRaw = JSON.parse(await readFile(cfg.utteranceMapPath, "utf-8")) as Record<string, UtteranceEntry>
  const utteranceMap = new Map(Object.entries(mapRaw))

  const censusRaw = JSON.parse(await readFile(join(cfg.artifactsDir, "census.json"), "utf-8")) as {
    per_card: Array<{ construct_id: string; bound?: number }>
  }
  const entrenchmentByConstruct = new Map<string, number>()
  for (const entry of censusRaw.per_card) {
    entrenchmentByConstruct.set(entry.construct_id, entry.bound ?? 0)
  }

  const planText = await readFile(join(cfg.artifactsDir, "fold_plan.jsonl"), "utf-8")
  const planRecordCount = planText.split("\n").filter((line) => line.trim().length > 0).length

  if (!cfg.corpusPath) throw new Error("bonfires-cxn: missing required config corpusPath (CXN_CORPUS)")
  if (!cfg.sessionTurnsPath) {
    throw new Error("bonfires-cxn: missing required config sessionTurnsPath (CXN_SESSION_TURNS)")
  }
  if (!cfg.embeddingsPath) throw new Error("bonfires-cxn: missing required config embeddingsPath (CXN_EMBEDDINGS)")

  const corpusRaw = JSON.parse(await readFile(cfg.corpusPath, "utf-8")) as Record<
    string,
    Omit<StatementEntry, "hash">
  >
  const statements = new Map<string, StatementEntry>()
  for (const [hash, entry] of Object.entries(corpusRaw)) statements.set(hash, { hash, ...entry })

  const turnsRaw = JSON.parse(await readFile(cfg.sessionTurnsPath, "utf-8")) as Record<string, Turn[]>
  const turns = new Map(Object.entries(turnsRaw))

  const embeddingsRaw = JSON.parse(await readFile(cfg.embeddingsPath, "utf-8")) as {
    model: string
    dim: number
    statements: Record<string, number[]>
    aggregates: { cxn: Record<string, number[]>; episode: Record<string, number[]> }
  }
  if (embeddingsRaw.model !== VOYAGE_MODEL) {
    throw new Error(
      `bonfires-cxn: embeddings model mismatch — file has "${embeddingsRaw.model}", expected "${VOYAGE_MODEL}"`
    )
  }
  const vectors = new Map(Object.entries(embeddingsRaw.statements))
  for (const hash of statements.keys()) {
    if (!vectors.has(hash)) {
      throw new Error(`bonfires-cxn: corpus hash ${hash} has no embedding vector — artifact drift`)
    }
  }
  const aggregates: Aggregates = {
    cxn: new Map(Object.entries(embeddingsRaw.aggregates.cxn)),
    episode: new Map(Object.entries(embeddingsRaw.aggregates.episode)),
  }

  return { utteranceMap, entrenchmentByConstruct, planRecordCount, statements, turns, vectors, aggregates }
}
