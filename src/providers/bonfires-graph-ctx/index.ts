import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
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
import { buildZepJudgePrompt } from "../zep/prompts"
import { buildKernelAnswerPrompt } from "../bonfires-kernel/prompts"

/**
 * `bonfires-graph-ctx`: serves the FROZEN conv-26 round-2 kernel-graph recipe to MemoryBench.
 *
 * Search = one POST to a local read-only shim (memory_kernel/scripts/proofs/conv26_round2_membench_shim.py) that
 * runs search_messages_v2 + build_answer_context over the already-built Neo4j group `g1-conv26-round2` and the
 * recipe's answer directive. The shim output is mapped into the SAME result shape the production `bonfires-kernel`
 * provider hands the answerer: one `cxn_context` carrier {lines, directive}. The answer prompt
 * (buildKernelAnswerPrompt -> buildAnswerPromptV3) and judge prompt (buildZepJudgePrompt) are the production ones,
 * untouched. Ingestion is SKIPPED: the graph was built beforehand; ingest()/awaitIndexing() record sessions only
 * and fail loudly if the shim does not attest the expected group.
 */
export interface GraphCtxConfig {
  shimUrl: string
  group: string
  metaPath?: string
}

export function loadGraphCtxConfig(
  env: Record<string, string | undefined> = process.env
): GraphCtxConfig {
  return {
    shimUrl: (env.GRAPHCTX_SHIM_URL || "http://127.0.0.1:9731").replace(/\/$/, ""),
    group: env.GRAPHCTX_GROUP || "g1-conv26-round2",
    metaPath: env.GRAPHCTX_META_PATH || undefined,
  }
}

export interface ShimHit {
  rank: number
  message_id: string
  text: string
  score: number
}
export interface ShimReply {
  context: string
  message_ids: string[]
  tokens: number
  directive: string | null
  frame: { shape: string; label: string | null }
  hits: ShimHit[]
  recipe: Record<string, unknown>
}

/** Pure shim-reply -> MemoryBench result items, in the layout production's `bonfires-kernel` gives the answerer:
 * the recipe context's `## Evidence` section (one message line each) becomes evidence items (`text`), rendered under
 * EVIDENCE; every other section (entity sheet, conversation windows, timeline) stays verbatim in the `cxn_context`
 * carrier's `lines`, rendered under CONTEXT WINDOW, with the directive on the carrier. Only the `## Evidence` header
 * line and blank lines inside that section are dropped; no content line is changed or lost. Ranked hits carry NO
 * `text`/`metadata.source_text`, so the prompt renders nothing for them; they exist for the retrieval-relevance side
 * metric and the kind-mix audit. */
export function mapShimReply(reply: ShimReply, shimHealth?: Record<string, unknown>): unknown[] {
  const hits = reply.hits.map((h) => ({
    kind: "kg_message",
    score: h.score,
    rank: h.rank,
    message_id: h.message_id,
    hit_text: h.text,
  }))
  const evidence: Array<Record<string, unknown>> = []
  const rest: string[] = []
  let inEvidence = false
  for (const line of reply.context.split("\n")) {
    if (line.startsWith("## ")) {
      inEvidence = line === "## Evidence"
      if (inEvidence) continue
    }
    if (inEvidence) {
      if (line.trim() !== "") evidence.push({ kind: "kg_evidence_line", text: line })
    } else {
      rest.push(line)
    }
  }
  const carrier: Record<string, unknown> = {
    kind: "cxn_context",
    lines: rest,
    kind_mix: { kg_message: hits.length, kg_evidence_line: evidence.length, cxn_context: 1 },
    recipe: {
      ...reply.recipe,
      frame: reply.frame,
      context_tokens: reply.tokens,
      context_message_ids: reply.message_ids,
      ingestion: "skipped-prebuilt-graph",
      ...(shimHealth ? { shim: shimHealth } : {}),
    },
  }
  if (reply.directive) carrier.directive = reply.directive
  return [...hits, ...evidence, carrier]
}

export class BonfiresGraphCtxProvider implements Provider {
  name = "bonfires-graph-ctx"
  concurrency = { default: 5, ingest: 1 }
  prompts = { answerPrompt: buildKernelAnswerPrompt, judgePrompt: buildZepJudgePrompt }

  private cfg: GraphCtxConfig | null = null
  private health: Record<string, unknown> | null = null
  private sessions = new Set<string>()
  private metaWritten = false

  async initialize(_config: ProviderConfig): Promise<void> {
    this.cfg = loadGraphCtxConfig()
    const response = await fetch(`${this.cfg.shimUrl}/health`)
    if (!response.ok)
      throw new Error(`bonfires-graph-ctx: shim /health failed (${response.status})`)
    const health = (await response.json()) as Record<string, any>
    const mismatches = health.frozen?.mismatches ?? {}
    if (health.group !== this.cfg.group || Object.keys(mismatches).length > 0) {
      throw new Error(
        `bonfires-graph-ctx: shim group/config mismatch (group=${health.group}, mismatches=${JSON.stringify(mismatches)})`
      )
    }
    this.health = {
      group: health.group,
      key_recipe_tag: health.key_recipe_tag,
      frozen_config_sha256: health.frozen?.frozen_config_sha256,
      census: health.census,
    }
    logger.info(
      `bonfires-graph-ctx: shim attests group ${health.group} census=${JSON.stringify(health.census)}`
    )
  }

  async ingest(sessions: UnifiedSession[], _options: IngestOptions): Promise<IngestResult> {
    for (const s of sessions) this.sessions.add(s.sessionId)
    return { documentIds: sessions.map((s) => s.sessionId), taskIds: [] }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    if (this.cfg?.metaPath && this.health && !this.metaWritten) {
      this.metaWritten = true
      mkdirSync(dirname(this.cfg.metaPath), { recursive: true })
      writeFileSync(
        this.cfg.metaPath,
        JSON.stringify(
          {
            provider: this.name,
            ingestion:
              "SKIPPED: graph g1-conv26-round2 was built before this run; ingest/indexing here only record session ids",
            sessions_seen: this.sessions.size,
            shim: this.health,
            at: new Date().toISOString(),
          },
          null,
          2
        ) + "\n",
        { flag: "wx" }
      )
    }
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  async search(query: string, _options: SearchOptions): Promise<unknown[]> {
    if (!this.cfg) throw new Error("bonfires-graph-ctx: provider not initialized")
    const response = await fetch(`${this.cfg.shimUrl}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    })
    if (!response.ok) {
      throw new Error(
        `bonfires-graph-ctx: shim /search failed (${response.status}): ${(await response.text()).slice(0, 300)}`
      )
    }
    return mapShimReply((await response.json()) as ShimReply, this.health ?? undefined)
  }

  async clear(containerTag: string): Promise<void> {
    logger.warn(`bonfires-graph-ctx: clear(${containerTag}) refused - prebuilt graph is read-only`)
  }
}

export default BonfiresGraphCtxProvider
