import { execSync } from "node:child_process"
import { writeFileSync } from "node:fs"
import { createHash } from "node:crypto"
import type {
  StackMessage,
  JobStatus,
  VectorSearchResult,
  KgDelveResult,
  ChunksSearchHit,
  HybridSearchResult,
} from "./types.js"

type FetchLike = typeof fetch

/**
 * Derive a deterministic 24-char hex ObjectId from a non-hex slug.
 * Uses SHA-1 of the slug string; takes first 24 hex chars.
 * This is purely deterministic — no randomness, same slug → same id across runs.
 */
function slugToObjectId(slug: string): string {
  return createHash("sha1").update(slug).digest("hex").slice(0, 24)
}

/** Return true if the string is already a valid 24-char hex ObjectId. */
function isHex24(s: string): boolean {
  return /^[0-9a-fA-F]{24}$/.test(s)
}

export interface BonfiresClientOptions {
  apiUrl: string
  apiKey?: string
  timeoutMs?: number
  fetchImpl?: FetchLike
}

export class BonfiresClient {
  private apiUrl: string
  private apiKey?: string
  private timeoutMs: number
  private fetchImpl: FetchLike

  constructor(opts: BonfiresClientOptions) {
    this.apiUrl = opts.apiUrl.replace(/\/$/, "")
    this.apiKey = opts.apiKey
    this.timeoutMs = opts.timeoutMs ?? 600_000
    this.fetchImpl = opts.fetchImpl ?? fetch
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" }
    if (this.apiKey) {
      h.Authorization = `Bearer ${this.apiKey}`
      h["X-API-Key"] = this.apiKey
    }
    return h
  }

  private async req<T>(
    method: "GET" | "POST" | "PUT" | "DELETE",
    path: string,
    body?: unknown,
    opts: { timeoutMs?: number } = {}
  ): Promise<T> {
    // ``timeoutMs`` on the constructor was previously stored but never
    // wired into fetch — long-running endpoints (resynthesize, chunks
    // grammar build on big bonfires) hit Bun's default fetch timeout
    // (~4m30s) mid-request. Apply the configured ceiling explicitly.
    const timeoutMs = opts.timeoutMs ?? this.timeoutMs
    const signal = timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : undefined
    const r = await this.fetchImpl(`${this.apiUrl}${path}`, {
      method,
      headers: this.headers(),
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    })
    if (!r.ok) {
      const text = await r.text().catch(() => "")
      throw new Error(`${method} ${path} failed ${r.status}: ${text}`)
    }
    return (await r.json()) as T
  }

  healthz(): Promise<{ status: string }> {
    return this.req("GET", "/healthz")
  }

  createAgent(args: { bonfireId: string; name: string }): Promise<{ id: string; name: string }> {
    return this.req("POST", "/agents", {
      name: args.name,
      username: args.name,
      bonfireId: args.bonfireId,
      isActive: true,
    })
  }

  async findOrCreateAgent(args: {
    bonfireId: string
    name: string
  }): Promise<{ id: string; name: string }> {
    try {
      return await this.createAgent(args)
    } catch (err) {
      const msg = String(err)
      if (!msg.includes("409")) throw err
      const list = await this.req<{ agents: Array<{ id: string; name: string }> }>("GET", "/agents")
      const match = list.agents.find((a) => a.name === args.name)
      if (!match) throw new Error(`409 on createAgent but no existing agent '${args.name}' found`)
      return match
    }
  }

  /** Append messages to an agent's stack. Backend validates 1-2 messages
   *  per request (paired user+assistant pattern), so for bulk we loop and
   *  send 2-at-a-time. All metadata on each StackMessage (including
   *  ``preserve_messages: true`` for LoCoMo-style JSON-structured episodes)
   *  flows through to the Mongo Message.metadata dict that stack_service
   *  reads at process time. */
  async stackAdd(agentId: string, messages: StackMessage[]): Promise<void> {
    for (let i = 0; i < messages.length; i += 2) {
      const batch = messages.slice(i, i + 2)
      await this.req("POST", `/agents/${agentId}/stack/add`, { messages: batch })
    }
  }

  stackProcess(agentId: string): Promise<{ task_id: string }> {
    return this.req("POST", `/agents/${agentId}/stack/process`)
  }

  /** Create an episode directly via Graphiti, bypassing stack_service's
   * LLM summarization of the transcript. Returns an arq task_id that can be
   * polled via waitForJob. Required for LoCoMo-style benchmarks where the
   * answer depends on specific quoted dialog text — stack_service's
   * _extract_episode_with_llm compresses dialog into a summary before
   * graphiti extraction, dropping verbatim claims.
   */
  /** GET /ontology/{bonfire_id} — read the current ontology (empty shape
   * if none exists). Used to merge fallback generic types into a
   * LLM-derived ontology before PUTting back. */
  getOntology(bonfireId: string): Promise<{
    bonfire_id: string
    entity_labels: Array<{
      name: string
      description: string
      labels: string[]
      fields: Record<string, unknown>
      parent_l1: string | null
    }>
    edge_labels: Array<{
      name: string
      description: string
      labels: string[]
      fields: Record<string, unknown>
      parent_l1: string | null
    }>
  }> {
    return this.req("GET", `/ontology/${bonfireId}`)
  }

  /** PUT /ontology/{bonfire_id} — set (overwrite) the bonfire's Ontology
   * Mongo doc. Use before running the indexing pipeline when you want a
   * specific curated entity-type set to guide graphiti's extraction,
   * rather than letting `derive_from_taxonomy` LLM-invent types. */
  setOntology(
    bonfireId: string,
    entityLabels: Array<{ name: string; description?: string; parent_l1?: string | null }>
  ): Promise<unknown> {
    return this.req("PUT", `/ontology/${bonfireId}`, {
      entity_labels: entityLabels.map((l) => ({
        name: l.name,
        description: l.description ?? "",
        labels: [l.name],
        fields: {},
        parent_l1: l.parent_l1 ?? null,
      })),
      edge_labels: [],
    })
  }

  /** Cascaded trimtab grammar search. Returns the walk text + expansion ids.
   * In cascaded mode (`rule` omitted) this walks origin → ...leaf and yields
   * a single generated string we can use to enrich a downstream vector
   * query — the whodunit "trimtabbed_vector" pattern: grammar as a prior
   * that injects role-relevant vocabulary into the query. */
  async searchGrammar(args: {
    bonfireId: string
    grammar: string
    query: string
    topK?: number
  }): Promise<{ mode: string; grammar: string; text?: string } & Record<string, unknown>> {
    return await this.req("POST", `/trimtabs/grammars/${args.bonfireId}/search`, {
      query: args.query,
      grammar: args.grammar,
      top_k: args.topK ?? 3,
      expand: false,
    })
  }

  createEpisodeDirect(args: {
    bonfireId: string
    name: string
    episodeBody: string
    referenceTime?: string
  }): Promise<{ success: boolean; task_id: string; status: string }> {
    const body: Record<string, unknown> = {
      bonfire_id: args.bonfireId,
      name: args.name,
      episode_body: args.episodeBody,
      source: "message",
      source_description: "memorybench_direct",
    }
    if (args.referenceTime) body.reference_time = args.referenceTime
    return this.req("POST", "/knowledge_graph/episode/create", body)
  }

  jobStatus(jobId: string): Promise<JobStatus> {
    return this.req("GET", `/jobs/${jobId}/status`)
  }

  async waitForJob(
    jobId: string,
    opts: { kind: string; timeoutSec?: number; pollIntervalSec?: number } = { kind: "job" }
  ): Promise<JobStatus> {
    const timeout = opts.timeoutSec ?? 3000
    const poll = opts.pollIntervalSec ?? 5
    const start = Date.now()
    await new Promise((r) => setTimeout(r, 2000))
    while (true) {
      const status = await this.jobStatus(jobId)
      if (status.state === "completed") return status
      if (status.state === "failed" || status.state === "cancelled") {
        throw new Error(`${opts.kind} job ${jobId} ended in ${status.state}: ${status.error ?? ""}`)
      }
      if ((Date.now() - start) / 1000 > timeout) {
        throw new Error(`${opts.kind} job ${jobId} exceeded ${timeout}s`)
      }
      await new Promise((r) => setTimeout(r, poll * 1000))
    }
  }

  startTaxonomy(bonfireId: string): Promise<{ job_id: string }> {
    // Optional GLiNER2 corpus-aggregate seed for the taxonomy workflow —
    // BONFIRES_TAXONOMY_SEED=engram triggers delve's engram_taxonomy_service
    // to pre-write top-K-per-label seed docs that the LLM refines.
    const body: Record<string, unknown> = { bonfire_id: bonfireId }
    const seed = process.env.BONFIRES_TAXONOMY_SEED
    if (seed) body.seed = seed
    return this.req("POST", "/trigger_taxonomy", body)
  }

  startLabelChunks(bonfireId: string): Promise<{ job_id: string }> {
    return this.req("POST", "/label_chunks", { bonfire_id: bonfireId })
  }

  /** Backfill the per-bonfire trimtab `chunks` grammar from Mongo. Runs
   * synchronously by default — LoCoMo-scale bonfires (<500 chunks) finish
   * in well under the ALB 60s window. Returns the completion stats.
   * Pass `sync: false` to enqueue as an arq task instead. */
  buildChunksGrammar(
    bonfireId: string,
    opts: { concurrency?: number; limit?: number; sync?: boolean } = {}
  ): Promise<{
    mode: "sync" | "async"
    bonfire_id?: string
    bonfire_name?: string
    chunks_read?: number
    chunks_with_summary?: number
    rules_inserted?: number
    rules_updated?: number
    error_count?: number
    task_id?: string
    status?: string
  }> {
    const body: Record<string, unknown> = {
      concurrency: opts.concurrency ?? 10,
      sync: opts.sync ?? true,
    }
    if (opts.limit !== undefined) body.limit = opts.limit
    return this.req("POST", `/trimtabs/grammars/${bonfireId}/chunks/build`, body)
  }

  /** Re-synthesize chunks using ontology-derived GLiNER labels. Must run
   * AFTER `buildGrammar` (so the Ontology doc exists with the LLM-
   * generated entity_labels + descriptions). Wipes session/preference/
   * topic chunks + their summaries, then re-runs SessionSynthesizer with
   * the ontology's labels merged into GLiNER's extraction set. Caller
   * must re-run `startSummaries` → `startLabelChunks` →
   * `buildChunksGrammar` → `propagateCascadeEmbeddings` after this so
   * the trimtab grammar is rebuilt against the new chunks. */
  resynthesizeChunks(
    bonfireId: string,
    opts: { sessionTitlePrefix?: string } = {}
  ): Promise<{
    bonfire_id: string
    ontology_labels_used: number
    session_docs: number
    chunks_deleted: number
    summaries_deleted: number
    chunks_created: number
    by_type: Record<string, number>
    ontology_span_hits: Record<string, number>
  }> {
    const body: Record<string, unknown> = {}
    if (opts.sessionTitlePrefix) body.session_title_prefix = opts.sessionTitlePrefix
    // Server streams whitespace heartbeats during the long synth so
    // Bun's 255s idle-fetch timeout doesn't fire mid-flight; the JSON
    // body is still parseable as one document. Per-call timeout still
    // bumped to 30 min as a hard ceiling in case the heartbeat path
    // fails for any reason.
    return this.req("POST", `/trimtabs/grammars/${bonfireId}/chunks/resynthesize`, body, {
      timeoutMs: 1_800_000,
    })
  }

  /** Write `metadata.cascade_embedding` on every chunk rule — HyperMem-lite
   * Phase B (v27i2). Must run after `buildChunksGrammar` so the offline
   * propagation sees the final embedded rules. Sync by default; the
   * math is in-memory + per-chunk metadata-only write, no re-embed. */
  propagateCascadeEmbeddings(
    bonfireId: string,
    opts: { lambda?: number; concurrency?: number; sync?: boolean } = {}
  ): Promise<{
    mode: "sync" | "async"
    bonfire_id?: string
    labels_seen?: number
    chunks_seen?: number
    chunks_updated?: number
    chunks_skipped_no_vector?: number
    chunks_skipped_no_label?: number
    unlabeled_bucket?: number
    error_count?: number
    task_id?: string
    status?: string
  }> {
    const body: Record<string, unknown> = {
      lambda: opts.lambda ?? 0.2,
      concurrency: opts.concurrency ?? 10,
      sync: opts.sync ?? true,
    }
    return this.req("POST", `/trimtabs/grammars/${bonfireId}/chunks/cascade-propagate`, body)
  }

  /** Tier 4 of the spaCy plan — build the dep-pattern index from
   * chunk.metadata.spacy_triples and MERGE canonical :Entity nodes +
   * verb-typed edges into Neo4j. Must run AFTER `resynthesizeChunks`
   * (which populates spacy_triples via spacy_enrich) and BEFORE
   * `createEpisodeDirect` so graphiti's add_episode dedup biases toward
   * the seeded entities. Sync = waits for build + seed to complete.
   */
  seedFromDepTree(
    bonfireId: string,
    opts: { sync?: boolean; minSupport?: number; maxEntities?: number; maxEdges?: number } = {}
  ): Promise<{
    mode: "sync" | "async"
    bonfire_id?: string
    triples_read?: number
    triples_skipped_low_support?: number
    triples_skipped_pronoun?: number
    triples_skipped_function_word?: number
    entities_seeded?: number
    edges_seeded?: number
    task_id?: string
    status?: string
  }> {
    return this.req("POST", `/trimtabs/grammars/${bonfireId}/chunks/seed-from-dep-tree`, {
      sync: opts.sync ?? true,
      min_support: opts.minSupport ?? 2,
      max_entities: opts.maxEntities ?? 200,
      max_edges: opts.maxEdges ?? 500,
    })
  }

  startLabeling(bonfireId: string, vectorThreshold = 0.7): Promise<{ job_id: string }> {
    return this.req("POST", "/labeling/hybrid", {
      bonfire_id: bonfireId,
      is_multi_label: false,
      taxonomy_run_id: null,
      vector_threshold: vectorThreshold,
    })
  }

  /**
   * Best-effort call to `/vector_store/setup`. Delve's endpoint is not
   * idempotent — it raises 500 if any collection already exists. Since
   * we can't wipe via HTTP (no delete endpoint is exposed), we swallow
   * the error: if the schema is already in place from a previous run,
   * update_labels will work regardless.
   */
  async setupVectorStore(): Promise<unknown> {
    try {
      const r = await this.fetchImpl(`${this.apiUrl}/vector_store/setup`, {
        method: "POST",
        headers: this.headers(),
      })
      if (r.ok) return r.json()
      return { status: "setup_skipped", httpStatus: r.status }
    } catch {
      return { status: "setup_skipped", reason: "network_error" }
    }
  }

  /**
   * Trigger VectorStoreService.update_labels_for_run. Creates a TaxonomyLabel
   * KG entity for each taxonomy without a uuid and saves the uuid back to
   * Mongo — the actual mechanism that populates Taxonomy.uuid. Takes the
   * TAXONOMY run_id (not a bonfire run_ref).
   */
  async updateLabels(bonfireId: string, runId: string): Promise<unknown> {
    const url =
      `${this.apiUrl}/vector_store/update_labels?bonfire_id=${encodeURIComponent(bonfireId)}` +
      `&run_id=${encodeURIComponent(runId)}`
    const r = await this.fetchImpl(url, { method: "POST", headers: this.headers() })
    if (!r.ok) {
      const text = await r.text().catch(() => "")
      throw new Error(`updateLabels failed ${r.status}: ${text}`)
    }
    return r.json()
  }

  buildGrammar(bonfireId: string, dryRun = false): Promise<unknown> {
    return this.req("POST", `/trimtabs/grammars/${bonfireId}/build`, { dry_run: dryRun })
  }

  buildOntology(
    bonfireId: string,
    opts: {
      linkToGraph?: boolean
      threshold?: number
      topNCap?: number
      extendGrammar?: string
      grammarMinMentions?: number
      grammarMinRelations?: number
      /** "cosine" (default) matches communities to ontology via embedding
       * similarity. "structural" counts ontology-label instances among
       * each community's :Entity members and links to the dominant label
       * — deterministic, noise-free, but requires graphiti ran with
       * ontology-guided entity_types. */
      linkMethod?: "cosine" | "structural"
    } = {}
  ): Promise<unknown> {
    const body: Record<string, unknown> = {
      entity_labels: null,
      link_to_graph: opts.linkToGraph ?? true,
    }
    if (opts.threshold !== undefined) body.threshold = opts.threshold
    if (opts.topNCap !== undefined) body.top_n_cap = opts.topNCap
    if (opts.linkMethod !== undefined) body.link_method = opts.linkMethod
    // Populate the `origin → taxonomy → ontology → community → entity` cascade
    // into a named grammar. Without this, the smart arm has no expansion set
    // and degrades to vector-only.
    if (opts.extendGrammar !== undefined) body.extend_grammar = opts.extendGrammar
    if (opts.grammarMinMentions !== undefined) body.grammar_min_mentions = opts.grammarMinMentions
    if (opts.grammarMinRelations !== undefined)
      body.grammar_min_relations = opts.grammarMinRelations
    return this.req("POST", `/ontology/${bonfireId}/build`, body)
  }

  async buildCommunities(bonfireId: string, sampleSize = 10): Promise<unknown> {
    // `sync=true` runs Leiden + community summaries inline. Required so the
    // subsequent ontology build sees real :Community nodes. Default-async is
    // fire-and-forget and produces an empty cascade for small local bonfires.
    const url = `${this.apiUrl}/knowledge_graph/communities/build?bonfire_id=${encodeURIComponent(
      bonfireId
    )}&sample_size=${sampleSize}&sync=true`
    const r = await this.fetchImpl(url, { method: "POST", headers: this.headers() })
    if (!r.ok) throw new Error(`buildCommunities failed ${r.status}`)
    return r.json()
  }

  /**
   * POST /trimtabs/grammars/{bonfireId}/chunks/search — hybrid retrieval over
   * the unified per-bonfire `chunks` grammar (trimtab HybridRetriever).
   * Returns verbatim chunk prose (`text`) plus the 1:1 LLM summary and
   * metadata. No KG calls; no cascade. Powers the `smart_chunks_only`
   * Engram-replication arm.
   *
   * `limit` maps to the route's `top_k`; `filterLabels` narrows the retrieval
   * pool to chunks tagged with any of the given taxonomy categories. Both
   * optional — the route accepts the omitted form (filter_labels=null).
   */
  async chunksSearch(args: {
    bonfireId: string
    query: string
    limit?: number
    filterLabels?: string[]
    nowDate?: string
    cascade?: boolean
  }): Promise<ChunksSearchHit[]> {
    return this.req<ChunksSearchHit[]>(
      "POST",
      `/trimtabs/grammars/${args.bonfireId}/chunks/search`,
      {
        query: args.query,
        top_k: args.limit ?? 10,
        filter_labels: args.filterLabels ?? null,
        now_date: args.nowDate ?? null,
        cascade: args.cascade ?? null,
      }
    )
  }

  async vectorSearch(args: {
    bonfireId: string
    query: string
    limit: number
  }): Promise<VectorSearchResult[]> {
    const raw = await this.req<{
      results: Array<{ id: string; properties: Record<string, unknown>; score: number | null }>
    }>("POST", "/vector_store/search", {
      bonfire_id: args.bonfireId,
      search_string: args.query,
      limit: args.limit,
    })
    return raw.results.map((r) => {
      const props = (r.properties ?? {}) as Record<string, unknown>
      return {
        id: r.id,
        text: (props.content as string | undefined) ?? "",
        doc_snippet: props.doc_snippet as string | undefined,
        score: r.score,
      }
    })
  }

  /**
   * Hub-traversal facts lane (delve POST /knowledge_graph/hub_traversal).
   * Returns enumerated entity facts for list-shaped questions ("what books
   * has Melanie read?") via a 1-2 hop walk through the IS_A hub structure.
   * Empty when the query is not a list-shape match.
   */
  hubTraversal(args: {
    bonfireId: string
    query: string
    maxDepth?: number
  }): Promise<{ facts: Array<{ text: string; kind: string; score: number | null }> }> {
    return this.req("POST", "/knowledge_graph/hub_traversal", {
      bonfire_id: args.bonfireId,
      query: args.query,
      max_depth: args.maxDepth ?? 2,
    })
  }

  kgDelve(args: {
    bonfireId: string
    query: string
    numResults: number
    centerNodeUuid?: string
    smart?: boolean
    searchRecipe?: string
    /** Per-scope BFS toggle (v0.6.1+): e.g. ['edges'] restricts multi-hop
     *  to the edge pool only, keeping node pool at bm25+cosine. Omit for
     *  default (BFS on both scopes when center is present). */
    bfsScopes?: Array<"nodes" | "edges">
    /** Per-scope cross-encoder rerank toggle (v0.6.1+): e.g. ['nodes','edges']
     *  keeps the recipe's reranker on those scopes, swaps others to RRF
     *  (cheap, no API call). Omit for default (recipe's own rerankers). */
    rerankScopes?: Array<"nodes" | "edges" | "episodes" | "communities">
  }): Promise<KgDelveResult> {
    const body: Record<string, unknown> = {
      bonfire_id: args.bonfireId,
      query: args.query,
      num_results: args.numResults,
    }
    if (args.centerNodeUuid) body.center_node_uuid = args.centerNodeUuid
    if (args.smart) body.smart = true
    if (args.searchRecipe) body.search_recipe = args.searchRecipe
    if (args.bfsScopes) body.bfs_scopes = args.bfsScopes
    if (args.rerankScopes) body.rerank_scopes = args.rerankScopes
    return this.req("POST", "/delve", body)
  }

  /**
   * v32 unified hybrid-retrieval endpoint (POST /search/hybrid).
   *
   * Composes chunks_search + kgDelve(raw|enriched|fanout) + hub_traversal
   * in one round-trip — replaces the bench's previous Promise.all over
   * three separate calls. Mode + seed gate live server-side; the response
   * carries chunks/entities/edges/hub_facts plus a debug payload that
   * tells us which lane(s) actually fired.
   *
   * Body keys are snake_case to match the Pydantic HybridSearchRequest.
   * See docs/plans/v32_unified_search_endpoint.md for the full design.
   */
  async hybridSearch(args: {
    bonfireId: string
    query: string
    // Sizing (server defaults: 20 / 10 / 20)
    topKChunks?: number
    topKEntities?: number
    topKFacts?: number
    // Mode + gating (server default: enriched_gated)
    mode?: "raw" | "enriched" | "enriched_gated" | "fanout"
    seedOverlapMin?: number
    seedSkipAggregates?: boolean
    // Sub-lane toggles (server default: all true)
    includeChunks?: boolean
    includeKg?: boolean
    includeHubFacts?: boolean
    // Optional in-server CE rerank (Phase 3 point reranker)
    rerank?: boolean
    rerankTopN?: number
    // Optional unified CE rerank over the merged pool of all four kinds
    // (chunks + entities + facts + hub_facts). When true, server populates
    // `unified_results` with a CE-ranked flat top-N; per-kind arrays in
    // the response stay populated unchanged for back-compat. Strictly
    // additive to `rerank` (KG-only) — when both are on the unified pass
    // runs LAST over the already-KG-reranked pool. Replaces the bench's
    // old post-hoc ceRerank round-trip.
    unifiedRerank?: boolean
    unifiedRerankTopN?: number
    // Standard delve knobs forwarded to kgDelve (server defaults wired)
    smart?: boolean
    searchRecipe?: string
    bfsScopes?: Array<"nodes" | "edges">
    rerankScopes?: Array<"nodes" | "edges" | "episodes" | "communities">
    nowDate?: string
    // Two-pass retrieval: when true the server runs chunks_search(top_k=1)
    // FIRST, builds an enriched query from the top-1 chunk's metadata
    // (text/spacy_triples/taxonomy_labels/l2_label/timestamp), and threads
    // both the enriched query AND the top-1's kg_entity_uuid (as an
    // additional graphiti BFS center) into the second-pass entities +
    // facts + chunks_search calls. Adds ~1-3s for the extra round-trip
    // but improves aggregate / multi-hop questions where the top chunk
    // surfaces a known entity that the raw query embedding misses.
    enrichFromTopChunk?: boolean
    // Run BOTH the raw and enriched (top-1-chunk-derived) query paths in
    // parallel and union the results before unified rerank. Combines
    // raw's temporal/single-hop precision with enriched's aggregate/
    // multi-hop expansion. Implies enrichFromTopChunk=true server-side;
    // ignores it if pass-0 returns no chunks. Costs +1 chunks_search
    // round-trip + 2x kgDelve cost; benefits from deduplication before
    // the final unified rerank.
    enrichedFanout?: boolean
    // Confidence-gate for enrichFromTopChunk / enrichedFanout. When >0,
    // the server only runs enrichment if the pass-0 top-1 chunk has at
    // least this many distinct extracted entities in
    // chunk.metadata.entities (summed across all type buckets). 0 (default)
    // = legacy: always enrich when the flag is on. Recommended: 3 — keeps
    // enrichment on multi-hop / aggregate queries (information-rich top-1)
    // and skips on vague single-hop pleasantries that dilute the enriched
    // query. Ignored when both enrichFromTopChunk and enrichedFanout are off.
    enrichMinCenters?: number
    // Entity-lane positive gate: only enrich when pass-0 top-1's
    // aux_lane_breakdown contains the 'entity' lane.
    enrichRequireEntityLane?: boolean
    // Community-cos inverted gate (>0): skip enrichment when pass-0
    // top-1's aux_lane_breakdown['community_cos'] exceeds threshold.
    enrichMaxCommunityCos?: number
    // NLP-based question-shape gate. When true, the server classifies
    // the query as "skip" (narrow factoid / modal / interpretive) or
    // "enrich" (open-ended) and overrides the chunk-metadata gates.
    // Default false = legacy chunk-metadata-only gating.
    enrichQuestionShapeGate?: boolean
    // Top-1 presearch: when the enrichment gate passes, re-run
    // chunks_search with the enriched query and use the result as the
    // final chunks pool. Restores v40-enrich's chunks-side enrichment
    // mechanism while keeping single-call behavior on gate-blocked
    // questions. Adds ~3.5s when gate passes, 0s when blocked.
    enrichChunksSearch?: boolean
    // MMR (Maximal Marginal Relevance) diversification: re-orders the
    // unified rerank pool to balance relevance with diversity. Useful
    // for multi-aspect / list questions where pure CE rerank clusters
    // near-duplicate chunks at the top. Requires unifiedRerank=true to
    // have any effect. lambda=1.0 → pure relevance (no-op); 0.0 → pure
    // diversity. Default 0.7.
    mmrDiversify?: boolean
    mmrLambda?: number
    // Per-lane MMR on facts/edges only — runs BEFORE the merged rerank
    // pool is built. Compresses near-duplicate facts (5+ variants of
    // "X pursues counseling" → 1-2 representatives) so diverse facts
    // get top-N slots. Independent of mmrDiversify; both can run.
    // Default lambda 0.6 (slightly diversity-leaning) since fact near-
    // duplication is the worst across kinds.
    mmrEdges?: boolean
    mmrEdgesLambda?: number
    // Disambiguated KG scopes: split delve into entity-only + fact-only
    // calls with scope-tuned query formulations (entities get
    // noun/category-rich query, facts get verb/temporal-rich query).
    // Costs +1 delve call per arm (~1-2s in parallel). Default false.
    disambiguatedKgScopes?: boolean
    // Gate-time presearch target: which trimtab symbol the top-1
    // enrichment lookup hits. "messages" (default) = single
    // conversation turns; "aggregates" = cross-session
    // preference/topic summaries (PreferenceHub/TopicHub UUIDs). For
    // broad/multi-aspect questions, aggregates produce richer
    // enrichment seeds. No effect when enrichFromTopChunk=false.
    presearchTarget?: "messages" | "aggregates"
    // Gate-time presearch source: which hydration mechanism produces
    // the enrichment gate's top-1 metadata. "chunks" (default) = full
    // chunks_search top-1 (~3.5s, full chunk + metadata blob).
    // "trimtab_cascade" = chunks-grammar cascade walk wrapped as a
    // single synthetic top-1 (sub-second, token-light, deterministic
    // on grammar). Orthogonal to presearchTarget — that field selects
    // the symbol within the "chunks" path only.
    presearchSource?: "chunks" | "trimtab_cascade"
    // Cascade-first parallel pipeline. When true the server runs the
    // trimtab cascade walk on the raw query first, builds an enriched
    // query from the walked text, and runs chunks_search + kg_entity +
    // kg_fact + hub in PARALLEL with the enriched query (bypasses the
    // pre-gate chunks_search + shape gate logic). Requires the bonfire's
    // primary_grammar to point at a multi-symbol cascade grammar; falls
    // back to legacy when cascade returns empty.
    cascadeFirstPipeline?: boolean
    // Slim the response — drops fields the bench doesn't read (chunk
    // summary + heavy chunk metadata, entity attributes, edge attributes,
    // unified_results id/metadata, debug elapsed_ms / overlap tokens /
    // gate counters) to cut FastAPI serialization cost. Default true:
    // none of the bench's smart_hybrid render path reads the dropped
    // fields. Set BONFIRES_HYBRID_RESPONSE_LEAN=0 to disable for debug.
    responseLean?: boolean
  }): Promise<HybridSearchResult> {
    const body: Record<string, unknown> = {
      bonfire_id: args.bonfireId,
      query: args.query,
    }
    if (args.topKChunks !== undefined) body.top_k_chunks = args.topKChunks
    if (args.topKEntities !== undefined) body.top_k_entities = args.topKEntities
    if (args.topKFacts !== undefined) body.top_k_facts = args.topKFacts
    if (args.mode !== undefined) body.mode = args.mode
    if (args.seedOverlapMin !== undefined) body.seed_overlap_min = args.seedOverlapMin
    if (args.seedSkipAggregates !== undefined) body.seed_skip_aggregates = args.seedSkipAggregates
    if (args.includeChunks !== undefined) body.include_chunks = args.includeChunks
    if (args.includeKg !== undefined) body.include_kg = args.includeKg
    if (args.includeHubFacts !== undefined) body.include_hub_facts = args.includeHubFacts
    if (args.rerank !== undefined) body.rerank = args.rerank
    if (args.rerankTopN !== undefined) body.rerank_top_n = args.rerankTopN
    if (args.unifiedRerank !== undefined) body.unified_rerank = args.unifiedRerank
    if (args.unifiedRerankTopN !== undefined) body.unified_rerank_top_n = args.unifiedRerankTopN
    if (args.smart !== undefined) body.smart = args.smart
    if (args.searchRecipe !== undefined) body.search_recipe = args.searchRecipe
    if (args.bfsScopes !== undefined) body.bfs_scopes = args.bfsScopes
    if (args.rerankScopes !== undefined) body.rerank_scopes = args.rerankScopes
    if (args.nowDate !== undefined) body.now_date = args.nowDate
    if (args.enrichFromTopChunk !== undefined) body.enrich_from_top_chunk = args.enrichFromTopChunk
    if (args.enrichedFanout !== undefined) body.enriched_fanout = args.enrichedFanout
    if (args.enrichMinCenters !== undefined) body.enrich_min_centers = args.enrichMinCenters
    if (args.enrichRequireEntityLane !== undefined)
      body.enrich_require_entity_lane = args.enrichRequireEntityLane
    if (args.enrichMaxCommunityCos !== undefined)
      body.enrich_max_community_cos = args.enrichMaxCommunityCos
    if (args.enrichQuestionShapeGate !== undefined)
      body.enrich_question_shape_gate = args.enrichQuestionShapeGate
    if (args.enrichChunksSearch !== undefined) body.enrich_chunks_search = args.enrichChunksSearch
    if (args.mmrDiversify !== undefined) body.mmr_diversify = args.mmrDiversify
    if (args.mmrLambda !== undefined) body.mmr_lambda = args.mmrLambda
    if (args.mmrEdges !== undefined) body.mmr_edges = args.mmrEdges
    if (args.mmrEdgesLambda !== undefined) body.mmr_edges_lambda = args.mmrEdgesLambda
    if (args.disambiguatedKgScopes !== undefined)
      body.disambiguated_kg_scopes = args.disambiguatedKgScopes
    if (args.presearchTarget !== undefined) body.presearch_target = args.presearchTarget
    if (args.presearchSource !== undefined) body.presearch_source = args.presearchSource
    if (args.cascadeFirstPipeline !== undefined)
      body.cascade_first_pipeline = args.cascadeFirstPipeline
    if (args.responseLean !== undefined) body.response_lean = args.responseLean
    return this.req<HybridSearchResult>("POST", "/search/hybrid", body)
  }

  ingestContent(args: {
    bonfireId: string
    content: string
    title: string
    metadata?: unknown
    messages?: Array<{
      speaker: string
      content: string
      timestamp?: string | null
      role?: string | null
      msg_id?: string | null
    }>
  }): Promise<unknown> {
    const body: Record<string, unknown> = {
      bonfire_id: args.bonfireId,
      content: args.content,
      title: args.title,
      metadata: args.metadata ?? null,
    }
    if (args.messages && args.messages.length > 0) {
      body.messages = args.messages
    }
    return this.req("POST", "/ingest_content", body)
  }

  startSummaries(bonfireId: string): Promise<{ job_id: string }> {
    return this.req("POST", "/generate_summaries", { bonfire_id: bonfireId })
  }

  createGrammar(args: {
    bonfireId: string
    grammar: string
    rules?: Record<string, unknown>
  }): Promise<unknown> {
    return this.req("POST", `/trimtabs/grammars/${args.bonfireId}`, {
      grammar: args.grammar,
      rules: args.rules ?? {},
    })
  }

  async seedGrammar(args: {
    bonfireId: string
    grammar: string
    rule: string
    kgQuery: string
    numEntities?: number
  }): Promise<unknown> {
    const url =
      `${this.apiUrl}/trimtabs/grammars/${args.bonfireId}/seed` +
      `?grammar=${encodeURIComponent(args.grammar)}` +
      `&rule=${encodeURIComponent(args.rule)}` +
      `&kg_query=${encodeURIComponent(args.kgQuery)}` +
      `&num_entities=${args.numEntities ?? 30}`
    const r = await this.fetchImpl(url, { method: "POST", headers: this.headers() })
    if (!r.ok) {
      const text = await r.text().catch(() => "")
      throw new Error(`seedGrammar failed ${r.status}: ${text}`)
    }
    return r.json()
  }

  /**
   * Ensure a bonfire document exists in MongoDB.
   *
   * If `bonfireId` is already a 24-char hex string it is used directly as the
   * ObjectId.  Otherwise a deterministic 24-char hex is derived from it via
   * SHA-1 (first 24 chars) so that each unique slug always maps to the same id.
   *
   * Tries `docker exec bonfires-mongo-1 mongosh ...` first.  If that command
   * is not available (no Docker, wrong container name) the bootstrap script is
   * written to `/tmp/bonfire-bootstrap-<id>.js` and an error is thrown that
   * tells the user to run it manually.
   */
  async ensureBonfire(args: {
    bonfireId: string
    name: string
    primaryGrammar?: string
  }): Promise<string> {
    const hexId = isHex24(args.bonfireId) ? args.bonfireId : slugToObjectId(args.bonfireId)
    const name = args.name
    const grammar = args.primaryGrammar ?? "locomo"

    const script = `
db.bonfires.updateOne(
  { _id: ObjectId("${hexId}") },
  { $setOnInsert: {
      type: "Bonfire",
      name: ${JSON.stringify(name)},
      purpose: "memorybench LoCoMo eval",
      is_public: false,
      taxonomy_refs: [],
      latest_taxonomy_run_refs: [],
      run_refs: [],
      createdAt: new Date(),
      updatedAt: new Date(),
      created_at: new Date(),
      updated_at: new Date(),
      max_agents: 10,
      max_episodes_per_agent: 1000,
      parent_bonfire_id: null,
      price_per_episode: null,
      primary_grammar: ${JSON.stringify(grammar)},
  }},
  { upsert: true }
);
`.trim()

    try {
      execSync(`docker exec bonfires-mongo-1 mongosh cannitos --quiet --eval '${script}'`, {
        stdio: "pipe",
      })
    } catch (err) {
      const scriptPath = `/tmp/bonfire-bootstrap-${hexId}.js`
      writeFileSync(scriptPath, script, "utf8")
      throw new Error(
        `ensureBonfire: docker exec failed (is bonfires-mongo-1 running?). ` +
          `Run this manually via mongosh: see ${scriptPath}\n` +
          `Original error: ${String(err)}`
      )
    }
    return hexId
  }
}
