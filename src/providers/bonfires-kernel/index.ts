import { readFile } from "node:fs/promises"
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
  // I2: gapmap.ts::extractAffordances reads `item.recipe.affordancesFired`/
  // `.fallback` off THIS enumerable item (JSON.stringify(items) is exactly
  // what gets persisted to results/<id>.json) — not off the non-enumerable
  // `.diagnostics` property search() attaches below, which JSON.stringify
  // silently drops. The wire recipe carries snake_case
  // `affordances_fired`; normalize to gapmap's camelCase key here (falling
  // back to an already-camelCase key, then `[]`, in case a future kernel
  // response sends it either way) and drop the snake_case duplicate so the
  // echoed recipe isn't carrying both spellings.
  if (envelope.recipe && typeof envelope.recipe === "object") {
    const wireRecipe = envelope.recipe as Record<string, unknown>
    const { affordances_fired, ...rest } = wireRecipe
    contextItem.recipe = {
      ...rest,
      affordancesFired: affordances_fired ?? wireRecipe.affordancesFired ?? [],
      fallback: envelope.fallback ?? false,
    }
  }
  return [...utterances, contextItem]
}

/** Wire shape of `GET /bonfires/{id}/kernel/state` (only the fields this
 * provider reads — see kernel_dto.py:KernelStateResponse). `census_digest`
 * lives under the `recipe` key alongside `cards_digest` (both come from
 * recipe_state_extra in graph-memory's recipe_support.py); it mirrors the
 * fold response's own top-level field (KernelIndexResponseWire below) — see
 * the doc comment on `checkCensusDigest` for why an absent field is treated
 * the same as a mismatch rather than "no check to do". */
interface KernelStateResponse {
  recipe?: {
    cards_digest?: string
    construct_universe_size?: number
    census_digest?: string | null
  } | null
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
  const base = referenceTime ? Date.parse(referenceTime) : undefined
  return session.messages.map((m, i) => {
    // The kernel-native fold (memory_kernel.fold.product._build_statement_corpus_and_turns)
    // indexes message dicts directly by `["text"]`/`["username"]`/["timestamp"]`
    // (no `.get()` fallback) — this shape is NOT the `{content, speaker}`
    // remapping the legacy `/search/memory-kernel/index` client path uses.
    //
    // Timestamps must be derived, never invented: a `Date.now()` fallback
    // here is nondeterministic — it bakes the wall-clock moment the bench
    // happened to run into statement content, which breaks the pinned
    // content-keyed extraction cache (a different timestamp -> a different
    // cache key -> silent live re-extraction) and makes two runs of the same
    // corpus non-reproducible. If a message has no timestamp of its own AND
    // the session carries no `metadata.date` to derive one from, that is a
    // data-completeness bug upstream (in extraction/sampling) that must be
    // fixed there, not papered over here.
    let timestamp = m.timestamp
    if (!timestamp) {
      if (base === undefined) {
        throw new Error(
          `bonfires-kernel: sessionToMessageBatch: message ${i} in session ` +
            `${session.sessionId} has no timestamp and session.metadata.date is ` +
            `unset — refusing to invent a wall-clock timestamp`
        )
      }
      timestamp = new Date(base + i * 120_000).toISOString()
    }
    return {
      text: m.content,
      username: m.speaker ?? m.role,
      timestamp,
      metadata: { ...(m.metadata ?? {}), session_id: session.sessionId },
    }
  })
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
    logger.info(`bonfires-kernel: config loaded for bonfire ${this.cfg.bonfireId}`)
  }

  // Sidecar/cards drift tripwire: when KERNELB_EXPECTED_CARDS_DIGEST is set,
  // the bench refuses to run against a bonfire whose fold artifacts were
  // built from a different grammar (cards) than the one this run expects —
  // otherwise a score delta could be attributable to a silently-drifted
  // recipe instead of the served pipeline. Public for testability (mirrors
  // bonfires-cxn's comprehendPreflight()).
  //
  // Called from foldIndex() AFTER the fold POST succeeds (and after the
  // census-digest tripwire check), NOT from initialize(). The cards digest
  // GET /kernel/state reads is derived from cards ∪ census mint events — on
  // a fresh bonfire, before any fold has run, that census doesn't exist yet
  // and the state route 503s (`recipe_cards_missing`). Preflighting in
  // initialize() therefore killed every fresh-bonfire run at startup, before
  // the fold that would have produced the very digest being checked.
  async preflightCardsDigest(): Promise<void> {
    const state = await this.fetchKernelState()
    this.checkCardsDigest(state)
  }

  private async fetchKernelState(): Promise<KernelStateResponse> {
    const cfg = this.requireConfig()
    const response = await this.fetchImpl(`${cfg.apiUrl}/bonfires/${cfg.bonfireId}/kernel/state`, {
      method: "GET",
      headers: { "X-Internal-Token": cfg.apiKey },
    })
    if (!response.ok) {
      throw new Error(`bonfires-kernel: preflight GET /kernel/state failed (${response.status})`)
    }
    return (await response.json()) as KernelStateResponse
  }

  private checkCardsDigest(state: KernelStateResponse): void {
    const cfg = this.requireConfig()
    const actual = state.recipe?.cards_digest
    if (actual !== cfg.expectedCardsDigest) {
      throw new Error(
        `bonfires-kernel: cards_digest mismatch — expected ${cfg.expectedCardsDigest}, got ${actual ?? "(none)"}`
      )
    }
  }

  // Skip-fold census tripwire (KERNELB_EXPECTED_CENSUS_DIGEST): in the normal
  // fold path this env var is checked against the fold POST's own response
  // (see foldIndex()). In skip-fold mode there is no fold POST to check
  // against, so — before this method existed — setting the env var in
  // skip-fold mode was silently inert: it was read into config but nothing
  // ever compared it to anything. That is exactly the loud-over-silent
  // violation this tripwire exists to prevent, and it hit in precisely the
  // mode the parity run used. Made functional here: compare the expected
  // digest against GET /kernel/state's `recipe.census_digest` (served from
  // the bonfire's census.json on disk), throwing on a mismatch OR when the
  // field is absent — absent means the bonfire has no census.json (never
  // folded), so graph-memory can't attest to which extraction produced the
  // pre-folded artifacts a skip-fold run is about to score against, which is
  // the same silent-drift risk as a mismatch and must fail loud, not pass.
  private checkCensusDigest(state: KernelStateResponse): void {
    const cfg = this.requireConfig()
    const actual = state.recipe?.census_digest
    if (actual !== cfg.expectedCensusDigest) {
      throw new Error(
        `bonfires-kernel: skip-fold census_digest mismatch — expected ${cfg.expectedCensusDigest}, ` +
          `got ${actual ?? "(absent from GET /kernel/state)"}`
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
      const cfg = this.requireConfig()
      if (cfg.skipFold) {
        // Search-only mode (KERNELB_SKIP_FOLD): the artifact bundle (fold
        // output + pinned normalization cache) was staged out-of-band, so
        // never POST /kernel/index here — that would burn a redundant LLM
        // extraction pass and, against a READONLY-pinned cache, silently
        // no-op the fold's own cache writes. Still preflight the cards
        // digest and/or census digest (whichever are configured) so a
        // mismatched pre-folded bonfire fails loudly instead of silently
        // scoring against the wrong artifact. One shared GET /kernel/state
        // fetch backs both checks — no second HTTP call.
        logger.info(
          "bonfires-kernel: KERNELB_SKIP_FOLD set — skipping POST /kernel/index " +
            "(search-only mode over a pre-folded bonfire)"
        )
        if (cfg.expectedCardsDigest || cfg.expectedCensusDigest) {
          const state = await this.fetchKernelState()
          if (cfg.expectedCardsDigest) {
            this.checkCardsDigest(state)
          }
          if (cfg.expectedCensusDigest) {
            this.checkCensusDigest(state)
          }
        }
      } else {
        await this.foldIndex()
      }
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
    // Pinned-batches mode (KERNELB_BATCHES_PATH): read the reference batch
    // JSON verbatim rather than reassembling from sessionsById. See the
    // `batchesPath` doc comment on KernelConfig for why byte-faithfulness
    // matters here — the parity corpus's cache key is content-derived, so
    // any reassembly drift (however small) silently falls through to live
    // re-extraction instead of hitting the pinned cache.
    const messageBatches = cfg.batchesPath
      ? await this.loadPinnedBatches(cfg.batchesPath)
      : [...this.sessionsById.values()].map(sessionToMessageBatch)
    const response = await this.fetchImpl(`${cfg.apiUrl}/bonfires/${cfg.bonfireId}/kernel/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Token": cfg.apiKey, "X-Permission": "write" },
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
    // Census-digest tripwire (KERNELB_EXPECTED_CENSUS_DIGEST): a loud failure
    // against silent extraction drift. If the fold's cache-key lookup missed
    // (e.g. because message_batches weren't byte-faithful to the pinned
    // reference), the kernel re-extracts live and produces a different
    // census_digest — better to blow up here than ship a mixed-fold
    // artifact dir that only fails later, per-query, with a KeyError.
    if (cfg.expectedCensusDigest && this.indexDiagnostics.census_digest !== cfg.expectedCensusDigest) {
      throw new Error(
        `bonfires-kernel: census_digest mismatch — expected ${cfg.expectedCensusDigest}, ` +
          `got ${this.indexDiagnostics.census_digest ?? "(none)"} (fold likely re-extracted ` +
          `instead of hitting the pinned cache)`
      )
    }
    // Cards-digest preflight runs here, post-fold — see the doc comment on
    // preflightCardsDigest() for why it can't run in initialize().
    if (cfg.expectedCardsDigest) {
      await this.preflightCardsDigest()
    }
  }

  private async loadPinnedBatches(path: string): Promise<Array<Array<Record<string, unknown>>>> {
    const raw = await readFile(path, "utf8")
    return JSON.parse(raw) as Array<Array<Record<string, unknown>>>
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
