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
import { loadKernelConfig, type KernelConfig } from "./config"
import { buildAnswerPromptV3 } from "../bonfires-cxn/retrieval2"

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>

/** One hit from the kernel `/search` envelope's `hits` array (Task 7 shape). */
export interface KernelSearchHit {
  uuid: string
  score: number
  text: string
  metadata?: Record<string, unknown>
}

/** The envelope shape `mapSearchEnvelope` consumes. Field is `results` here
 * (matching the spec's test fixture) even though the wire response from
 * `POST /kernel/search` names the array `hits` — `search()` below is
 * responsible for that field-name translation before calling this pure
 * function. */
export interface KernelSearchEnvelope {
  results: KernelSearchHit[]
  context_lines: string[]
  directive: string | null
  recipe: Record<string, unknown> | null
  fallback: boolean
}

/** Pure envelope -> bench result items mapping. NO retrieval logic here —
 * every score/rank/text was already decided server-side by the kernel's
 * `/kernel/search` route; this only reshapes the wire envelope into the
 * `{text, kind, score, metadata}` item shape the harness (and
 * `buildAnswerPromptV3`) expect, plus one `cxn_context` item carrying the
 * hydration lines + answer directive. The `directive` key is OMITTED
 * (not set to `undefined`) when the envelope's directive is null/absent —
 * `buildAnswerPromptV3` and the harness's checkpoint slimming both branch on
 * key presence, not truthiness. */
export function mapSearchEnvelope(envelope: KernelSearchEnvelope): unknown[] {
  const utterances = envelope.results.map((hit) => ({
    text: hit.text,
    kind: "cxn_utterance" as const,
    score: hit.score,
    metadata: hit.metadata ?? {},
  }))
  const contextItem: Record<string, unknown> = {
    kind: "cxn_context",
    lines: envelope.context_lines,
  }
  if (envelope.directive !== null && envelope.directive !== undefined) {
    contextItem.directive = envelope.directive
  }
  return [...utterances, contextItem]
}

/** Wire shape of `GET /bonfires/{id}/kernel/state` (only the fields this
 * provider reads — see kernel_dto.py:KernelStateResponse). */
interface KernelStateResponse {
  recipe?: { cards_digest?: string; construct_universe_size?: number } | null
}

/** Wire shape of `POST /bonfires/{id}/kernel/search` (see
 * kernel_dto.py:KernelSearchResponse). Field is `hits`, not `results` —
 * translated into `KernelSearchEnvelope` by `search()` below. */
interface KernelSearchResponseWire {
  hits?: KernelSearchHit[]
  context_lines?: string[]
  directive?: string | null
  recipe?: Record<string, unknown> | null
  fallback?: boolean | null
}

/** Wire shape of `POST /bonfires/{id}/kernel/index` for the cxn-fold path
 * (metadata.cxn_fold=true) — see kernel_dto.py:KernelIndexResponse. */
interface KernelIndexResponseWire {
  census_digest?: string | null
  statement_count?: number | null
  construct_universe?: string[]
}

function sessionToMessageBatch(session: UnifiedSession): Array<Record<string, unknown>> {
  const referenceTime = session.metadata?.date as string | undefined
  const base = referenceTime ? Date.parse(referenceTime) : Date.now()
  return session.messages.map((m, i) => ({
    // The kernel-native fold (memory_kernel.fold.product._build_statement_corpus_and_turns)
    // indexes message dicts directly by `["text"]`/`["username"]`/["timestamp"]`
    // (no `.get()` fallback) — this shape is NOT the `{content, speaker}`
    // remapping the legacy `/search/memory-kernel/index` client path uses.
    text: m.content,
    username: m.speaker ?? m.role,
    timestamp: m.timestamp ?? new Date(base + i * 120_000).toISOString(),
    metadata: { ...(m.metadata ?? {}), session_id: session.sessionId },
  }))
}

export class BonfiresKernelProvider implements Provider {
  name = "bonfires-kernel"
  concurrency = { default: 5, ingest: 1 }
  prompts = { answerPrompt: buildAnswerPromptV3 }

  private cfg: KernelConfig | null
  private fetchImpl: FetchLike

  // Accumulator of unique sessions seen during ingest(). The kernel-native
  // fold is one POST for the whole corpus (no per-session drain via stacks —
  // see the task brief's ambiguity resolution), so ingest() only gathers
  // sessions; awaitIndexing() does the actual index call, once, on its first
  // invocation (mirroring the `indexingDone` guard the sibling `bonfires`
  // provider uses for the same "orchestrator calls awaitIndexing per
  // question" reason).
  private sessionsById = new Map<string, UnifiedSession>()
  private indexingDone = false
  private indexDiagnostics: KernelIndexResponseWire | null = null

  // Test constructor: inject cfg + fetch. Production path: no-arg + initialize().
  constructor(cfg?: KernelConfig, fetchImpl?: FetchLike) {
    this.cfg = cfg ?? null
    this.fetchImpl = fetchImpl ?? (globalThis.fetch as FetchLike)
  }

  async initialize(_config: ProviderConfig): Promise<void> {
    if (!this.cfg) this.cfg = loadKernelConfig()
    if (this.cfg.expectedCardsDigest) {
      await this.preflightCardsDigest()
    }
    logger.info(`bonfires-kernel: preflight OK for bonfire ${this.cfg.bonfireId}`)
  }

  // Sidecar/cards drift tripwire: when KERNELB_EXPECTED_CARDS_DIGEST is set,
  // the bench refuses to run against a bonfire whose fold artifacts were
  // built from a different grammar (cards) than the one this run expects —
  // otherwise a score delta could be attributable to a silently-drifted
  // recipe instead of the served pipeline. Public for testability (mirrors
  // bonfires-cxn's comprehendPreflight()).
  async preflightCardsDigest(): Promise<void> {
    const cfg = this.requireConfig()
    const response = await this.fetchImpl(`${cfg.apiUrl}/bonfires/${cfg.bonfireId}/kernel/state`, {
      method: "GET",
      headers: { "X-Internal-Token": cfg.apiKey },
    })
    if (!response.ok) {
      throw new Error(`bonfires-kernel: preflight GET /kernel/state failed (${response.status})`)
    }
    const state = (await response.json()) as KernelStateResponse
    const actual = state.recipe?.cards_digest
    if (actual !== cfg.expectedCardsDigest) {
      throw new Error(
        `bonfires-kernel: cards_digest mismatch — expected ${cfg.expectedCardsDigest}, got ${actual ?? "(none)"}`
      )
    }
  }

  async ingest(sessions: UnifiedSession[], _options: IngestOptions): Promise<IngestResult> {
    const documentIds: string[] = []
    for (const session of sessions) {
      if (!this.sessionsById.has(session.sessionId)) {
        this.sessionsById.set(session.sessionId, session)
      }
      documentIds.push(session.sessionId)
    }
    return { documentIds, taskIds: [] }
  }

  async awaitIndexing(
    result: IngestResult,
    _containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    if (!this.indexingDone) {
      await this.foldIndex()
      this.indexingDone = true
    }
    onProgress?.({
      completedIds: result.documentIds,
      failedIds: [],
      total: result.documentIds.length,
    })
  }

  private async foldIndex(): Promise<void> {
    const cfg = this.requireConfig()
    const messageBatches = [...this.sessionsById.values()].map(sessionToMessageBatch)
    const response = await this.fetchImpl(`${cfg.apiUrl}/bonfires/${cfg.bonfireId}/kernel/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Token": cfg.apiKey },
      body: JSON.stringify({
        actor_id: cfg.actorId,
        mode: "upsert",
        message_batches: messageBatches,
        metadata: { cxn_fold: true },
      }),
    })
    if (!response.ok) {
      throw new Error(`bonfires-kernel: POST /kernel/index failed (${response.status})`)
    }
    this.indexDiagnostics = (await response.json()) as KernelIndexResponseWire
    logger.info(
      `bonfires-kernel: cxn fold indexed ${this.indexDiagnostics.statement_count ?? "?"} statements ` +
        `across ${messageBatches.length} sessions (census_digest=${this.indexDiagnostics.census_digest ?? "?"})`
    )
  }

  async search(query: string, _options: SearchOptions): Promise<unknown[]> {
    const cfg = this.requireConfig()
    const response = await this.fetchImpl(`${cfg.apiUrl}/bonfires/${cfg.bonfireId}/kernel/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Token": cfg.apiKey },
      body: JSON.stringify({ query, top_k: 20 }),
    })
    if (!response.ok) {
      throw new Error(`bonfires-kernel: POST /kernel/search failed (${response.status})`)
    }
    const wire = (await response.json()) as KernelSearchResponseWire
    const envelope: KernelSearchEnvelope = {
      results: wire.hits ?? [],
      context_lines: wire.context_lines ?? [],
      directive: wire.directive ?? null,
      recipe: wire.recipe ?? null,
      fallback: wire.fallback ?? false,
    }
    const items = mapSearchEnvelope(envelope)
    // Recipe echo -> per-question diagnostics side channel (mirrors the
    // `(results as Record<string, unknown>).diagnostics` convention the
    // orchestrator's search phase already reads via
    // extractSearchDiagnosticsForCheckpoint). Non-enumerable so it never
    // leaks into JSON.stringify(items) / the per-question result dump —
    // arrays only serialize their indices regardless, this just keeps
    // Object.keys/for-in clean too.
    Object.defineProperty(items, "diagnostics", {
      value: { recipe: envelope.recipe, fallback: envelope.fallback, index: this.indexDiagnostics },
      enumerable: false,
    })
    return items
  }

  async clear(containerTag: string): Promise<void> {
    // The kernel-native fold is the system under test, not run state —
    // never delete it out from under a bench run.
    logger.warn(
      `bonfires-kernel: clear(${containerTag}) refused — cxn-fold bonfire is read-only for this provider`
    )
  }

  private requireConfig(): KernelConfig {
    if (!this.cfg) throw new Error("bonfires-kernel: provider not initialized")
    return this.cfg
  }
}

export default BonfiresKernelProvider
