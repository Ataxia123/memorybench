import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { CxnConfig } from "./config"

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

  return { utteranceMap, entrenchmentByConstruct, planRecordCount }
}
