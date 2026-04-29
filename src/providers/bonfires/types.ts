export interface DelveAgent {
  id: string
  name: string
  bonfireId: string
}

export interface StackMessage {
  id: string
  text: string
  userId: string
  chatId: string
  timestamp: string // ISO 8601
  role: string // "user" / "assistant" / "agent"
  username?: string // display name of speaker (e.g., "Caroline") — optional
  /** Stack V2 — partitions a multi-session stack into one episode batch
   *  per session. When omitted, delve falls back to chatId so V1 ingestion
   *  is unchanged. Set this when pushing N sessions through one stackProcess
   *  call (BONFIRES_STACK_V2=1 mode). */
  sessionId?: string
  /** Per-message metadata propagated to Mongo's Message.metadata dict.
   * Recognized keys (read by delve):
   *   preserve_messages: true  → opt-in for JSON-structured episode output
   *                              (stack_service persists raw messages in
   *                              structured_content + sets source="json"). */
  metadata?: Record<string, unknown>
}

export interface JobStatus {
  state: "pending" | "running" | "completed" | "failed" | "cancelled"
  error?: string
  // Legacy top-level result — used by some callers. The actual workflow
  // output usually lives under metadata.result instead.
  result?: unknown
  metadata?: {
    job_uuid?: string
    result?: Record<string, unknown>
  }
}

export interface VectorSearchResult {
  text: string
  score: number | null
  id?: string
  doc_snippet?: string
}

export interface KgDelveEdge {
  fact: string
  score?: number
  uuid?: string
  valid_at?: string | null
}

export interface KgDelveEntity {
  name: string
  summary?: string
  uuid?: string
  labels?: string[]
}

export interface KgDelveEpisode {
  name: string
  content?: string
  summary?: string
  valid_at?: string | null
  uuid?: string
}

export interface KgDelveCommunity {
  name?: string
  summary?: string
  uuid?: string
}

export interface KgDelveResult {
  edges?: KgDelveEdge[]
  nodes?: Array<{ name: string; summary?: string; uuid?: string }>
  entities?: KgDelveEntity[]
  episodes?: KgDelveEpisode[]
  communities?: KgDelveCommunity[]
}

/** Hit from `POST /trimtabs/grammars/{bonfire_id}/chunks/search`.
 * One hit = one ChunkRule surfaced by HybridRetriever over the per-bonfire
 * `chunks` grammar. `text` is the verbatim prose we show the answerer;
 * `summary` is the 1:1 LLM summary that was embedded (summary#labels).
 * `metadata` is passthrough — fields depend on ingest source (doc-chunk vs
 * stack-message) and whether lazy KG promotion has fired for this chunk. */
export interface ChunksSearchHit {
  id: string
  text: string
  summary: string | null
  score: number
  metadata: {
    categories?: string[]
    kg_entity_uuid?: string | null
    kg_edge_uuids?: string[]
    session_id?: string
    msg_id?: string
    speaker?: string
    timestamp?: string
    retrieved_count?: number
    last_accessed?: string
    [key: string]: unknown
  }
}

/** One ranked item from /search/hybrid when unified_rerank=True.
 * Mirrors UnifiedRerankItem on the delve side. The bench renders these
 * directly as SearchHit when present, replacing per-kind merge + the
 * legacy bench-side ceRerank round-trip. */
export interface UnifiedRerankItem {
  text: string
  score: number
  kind: "chunk" | "entity" | "fact" | "hub_fact"
  id: string
  metadata: Record<string, unknown>
}

/** Response shape from `POST /search/hybrid` (v32 unified endpoint).
 * Composes chunks + entities + edges + hub_facts in one payload.
 * Mirrors HybridSearchResponse on the delve side. When the request sets
 * unified_rerank=True the server additionally populates `unified_results`
 * with a CE-ranked flat top-N across all kinds; per-kind arrays remain
 * populated unchanged for back-compat. */
export interface HybridSearchResult {
  chunks: ChunksSearchHit[]
  entities: KgDelveEntity[]
  edges: KgDelveEdge[]
  hub_facts: Array<{ text: string; kind: string; score: number | null }>
  unified_results: UnifiedRerankItem[] | null
  debug: {
    mode: "raw" | "enriched" | "enriched_gated" | "fanout"
    kg_query_used: string | null
    seed_chunk_id: string | null
    seed_accepted: boolean
    seed_overlap_tokens: string[]
    elapsed_ms: Record<string, number>
  }
}

export type BonfiresArm =
  | "vector"
  | "graph"
  | "smart_graph"
  | "smart"
  | "smart_full"
  | "smart_naked"
  | "smart_chunks_only"
  | "smart_unified"
  | "smart_cascade"
  | "smart_hybrid"
  | "zep"

export interface BonfiresConfig {
  apiUrl: string
  apiKey?: string
  arm: BonfiresArm
  bonfireId: string
}
