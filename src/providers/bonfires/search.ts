import type { BonfiresClient } from "./client.js"
import type { BonfiresConfig, KgDelveResult } from "./types.js"

/** Set of entity labels dropped from the rerank pool by smart_hybrid.
 * PreferenceHub is the per-speaker aggregate that unions ALL preferences
 * for a speaker, giving it max BGE cosine to any question mentioning the
 * speaker + any topic. Other aggregates (PreferenceRecord/TopicRecord) are
 * per-session content-rich and stay in the pool. smart_unified inlines the
 * same set in-case for byte-for-byte compatibility with prior benches. */
const HYBRID_HUB_ENTITY_LABELS_TO_DROP = new Set(["PreferenceHub"])

export interface SearchHit {
  text: string
  score: number | null
  /** Scope tag lets the answer prompt categorize the hit into zep-style
   *  <FACTS> vs <ENTITIES> sections. "fact" = edges (relationship claims
   *  with valid_at), "entity" = nodes (name + summary). Optional — hits
   *  without a kind fall through as "other" (treated as facts). */
  kind?: "fact" | "claim" | "entity" | "episode" | "community" | "chunk" | "answer_hint"
  metadata?: Record<string, unknown>
}

export function flattenFacts(result: KgDelveResult): SearchHit[] {
  const edges = result.edges ?? []
  return edges.map((e) => ({ text: e.fact, score: e.score ?? null, kind: "fact" as const }))
}

function orderedHypermemHits(
  outputType: string,
  hits: {
    topics: SearchHit[]
    episodes: SearchHit[]
    facts: SearchHit[]
    evidence: SearchHit[]
  }
): SearchHit[] {
  const bits = /^[01]{3}$/.test(outputType) ? outputType : "011"
  const order = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER ?? "canonical"
  const includeEvidence = process.env.BONFIRES_HYPERMEM_INCLUDE_EVIDENCE !== "0"
  const out: SearchHit[] = []
  if (order === "score") {
    if (bits[0] === "1") out.push(...hits.topics)
    if (bits[1] === "1") out.push(...hits.episodes)
    if (bits[2] === "1") out.push(...hits.facts)
    if (includeEvidence) out.push(...hits.evidence)
    return out.sort((a, b) => (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY))
  }
  if (order === "facts_first") {
    if (bits[2] === "1") out.push(...hits.facts)
    if (includeEvidence) out.push(...hits.evidence)
    if (bits[1] === "1") out.push(...hits.episodes)
    if (bits[0] === "1") out.push(...hits.topics)
    return out
  }
  if (bits[0] === "1") out.push(...hits.topics)
  if (bits[1] === "1") out.push(...hits.episodes)
  if (bits[2] === "1") out.push(...hits.facts)
  if (includeEvidence) out.push(...hits.evidence)
  return out
}

const HYPERMEM_METADATA_SCALAR_KEYS = [
  "id",
  "topic_id",
  "source_type",
  "source_episode_id",
  "temporal",
  "timestamp",
  "point_score",
  "confidence",
  "importance_weight",
  "graph_edge_uuid",
  "fact_uuid",
  "episode_uuid",
  "subject",
  "title",
] as const

const HYPERMEM_METADATA_LIST_KEYS = [
  "episode_ids",
  "topic_ids",
  "keywords",
  "query_patterns",
  "potential_queries",
  "participants",
  "user_ids",
  "topic_route",
] as const

function slimHypermemMetadata(data: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!data) return {}
  const out: Record<string, unknown> = {}
  for (const key of HYPERMEM_METADATA_SCALAR_KEYS) {
    const value = data[key]
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      out[key] = value
    }
  }
  for (const key of HYPERMEM_METADATA_LIST_KEYS) {
    const value = data[key]
    if (Array.isArray(value)) {
      out[key] = value
        .filter((item) => typeof item === "string" || typeof item === "number" || typeof item === "boolean")
        .slice(0, key === "keywords" ? 24 : 12)
    }
  }
  return out
}

export async function armSearch(args: {
  client: Pick<
    BonfiresClient,
    "vectorSearch" | "kgDelve" | "chunksSearch" | "hubTraversal" | "hybridSearch" | "hypermemSearch"
  >
  query: string
  config: BonfiresConfig
  /** ISO ``YYYY-MM-DD`` reference "now" for temporal auxiliary ranking in
   * chunks_search. Enables the ``"yesterday"`` / ``"a week ago"`` signal.
   * For LoCoMo: max session date per conversation is the natural anchor. */
  nowDate?: string
}): Promise<SearchHit[]> {
  const { client, query, config, nowDate } = args
  try {
    switch (config.arm) {
      case "hypermem": {
        const outputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE ?? "011"
        const r = await client.hypermemSearch({
          bonfireId: config.bonfireId,
          query,
          profile: process.env.BONFIRES_HYPERMEM_PROFILE ?? "nlp_taxonomy_v1",
          initialCandidates: parseInt(process.env.BONFIRES_HYPERMEM_INITIAL_CANDIDATES ?? "100", 10),
          topicTopK: parseInt(process.env.BONFIRES_HYPERMEM_TOPIC_TOP_K ?? "15", 10),
          episodeTopK: parseInt(process.env.BONFIRES_HYPERMEM_EPISODE_TOP_K ?? "20", 10),
          factTopK: parseInt(process.env.BONFIRES_HYPERMEM_FACT_TOP_K ?? "30", 10),
          outputType,
          useReranker: process.env.BONFIRES_HYPERMEM_RERANKER !== "0",
        })
        const facts = (r.facts ?? [])
          .map((fact) => {
            const data = fact.data ?? {}
            const content = String(data.content ?? data.fact ?? data.summary ?? "").trim()
            // Delve's HyperMem formatter only treats `temporal` as event time.
            // `timestamp` is node bookkeeping for grammar-derived facts and can
            // be ingestion time; rendering it poisoned LoCoMo answers with the
            // benchmark run date.
            const timestamp = data.temporal
            return { fact, content, timestamp }
          })
          .filter((item) => item.content)
          .map(({ fact, content, timestamp }) => ({
            text:
              timestamp
                ? `[FACT] ${content} (event_time: ${String(timestamp)})`
                : `[FACT] ${content}`,
            score: fact.score ?? null,
            kind: "fact" as const,
            metadata: {
              source: fact.source,
              hypermem: slimHypermemMetadata(fact.data),
            },
          }))
        const episodes = (r.episodes ?? [])
          .map((episode) => {
            const data = episode.data ?? {}
            const content = String(data.summary ?? data.subject ?? data.episode_description ?? "").trim()
            const timestamp = data.timestamp
            return { episode, content, timestamp }
          })
          .filter((item) => item.content)
          .map(({ episode, content, timestamp }) => ({
            text:
              timestamp
                ? `[EPISODE] ${content} (event_time: ${String(timestamp)})`
                : `[EPISODE] ${content}`,
            score: episode.score ?? null,
            kind: "episode" as const,
            metadata: {
              hypermem: slimHypermemMetadata(episode.data),
            },
          }))
        const topics = (r.topics ?? [])
          .map((topic) => {
            const data = topic.data ?? {}
            const title = String(data.title ?? "").trim()
            const summary = String(data.summary ?? "").trim()
            return { topic, title, summary }
          })
          .filter((item) => item.title || item.summary)
          .map(({ topic, title, summary }) => ({
            text: `[TOPIC] ${title ? `${title}: ` : ""}${summary}`,
            score: topic.score ?? null,
            kind: "community" as const,
            metadata: {
              hypermem: slimHypermemMetadata(topic.data),
            },
          }))
        const evidence = (r.evidence ?? [])
          .map((item) => {
            const data = item.data ?? {}
            const content = String(data.content ?? data.fact ?? data.summary ?? "").trim()
            const timestamp = data.temporal
            return { item, content, timestamp }
          })
          .filter((item) => item.content)
          .map(({ item, content, timestamp }) => ({
            text:
              timestamp
                ? `[EVIDENCE] ${content} (event_time: ${String(timestamp)})`
                : `[EVIDENCE] ${content}`,
            score: item.score ?? null,
            kind: "fact" as const,
            metadata: {
              source: item.source,
              hypermem: slimHypermemMetadata(item.data),
            },
          }))
        return orderedHypermemHits(outputType, { topics, episodes, facts, evidence })
      }
      case "smart_chunks_only": {
        // Engram replication — pure hybrid retrieval over the trimtab chunks
        // grammar. No KG calls, no cascade, no vector_store. Tests the
        // retrieval ceiling on a single-store setup and exercises the
        // zero-extraction cold-start path (works before any chunk has been
        // promoted to graphiti).
        const hits = await client.chunksSearch({
          bonfireId: config.bonfireId,
          query,
          limit: 20,
          nowDate,
        })
        return hits.map((h) => ({
          text: h.text,
          score: h.score,
          kind: "chunk" as const,
        }))
      }
      case "smart_full": {
        // All four scopes from COMBINED_HYBRID_SEARCH_CROSS_ENCODER
        // (graphiti's default 4-scope recipe). Single smart=true call
        // means the cascade runs once and seeds BFS from walk UUIDs for
        // node + edge scopes. Episode + community scopes are now surfaced
        // to the answer context (previously discarded). Costs 4× BGE on
        // one GPU (~4s/query) but exercises the full retrieval surface.
        //   - entities: "{name}: {summary}" (20)
        //   - edges:    "{fact} (event_time: {valid_at})" (20)
        //   - episodes: "{speaker line / content}" (up to 20 dedup'd)
        //   - communities: "{name}: {summary}" (whatever the reranker returns)
        // Benchmark knobs (see `case "smart"` below for semantics).
        const bfsScopesEnvFull = process.env.BONFIRES_BFS_SCOPES
        const rerankScopesEnvFull = process.env.BONFIRES_RERANK_SCOPES
        const bfsScopesFull = bfsScopesEnvFull
          ? (bfsScopesEnvFull
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges">)
          : undefined
        const rerankScopesFull = rerankScopesEnvFull
          ? (rerankScopesEnvFull
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges" | "episodes" | "communities">)
          : undefined
        const r = await client.kgDelve({
          bonfireId: config.bonfireId,
          query,
          numResults: 20,
          smart: true,
          bfsScopes: bfsScopesFull,
          rerankScopes: rerankScopesFull,
        })
        const entities = (r.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }))
        const facts = (r.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }))
        // v25+ bonfires: stack_service stores episodes as a JSON object
        // with shape { name, content, updates, messages }. The `content`
        // field is a ~300-char LLM summary — the lean text we want in the
        // answer context. Without this helper we were JSON-stringifying
        // the full object (name + content + 5 updates + 18 messages) and
        // shipping 8.5 KB per episode × ~27 episodes/query ≈ 60 k tokens.
        //
        // Fallback chain:
        //   1. `ep.summary` string (older bonfire shape)
        //   2. `content.content` string (v25+ summary prose) ← target
        //   3. `content.summary` string (hypothetical future shape)
        //   4. `ep.content` string (pre-v25 raw message dump)
        //   5. `content.name` string (headline; last resort short form)
        //   6. "" (skip)
        const toText = (ep: { content?: unknown; summary?: string }): string => {
          if (typeof ep.summary === "string" && ep.summary.length > 0) return ep.summary
          const raw = ep.content
          if (raw && typeof raw === "object") {
            const r = raw as Record<string, unknown>
            if (typeof r.content === "string" && r.content.length > 0) return r.content
            if (typeof r.summary === "string" && r.summary.length > 0) return r.summary
            if (typeof r.name === "string" && r.name.length > 0) return r.name
            return ""
          }
          if (typeof raw === "string" && raw.length > 0) return raw
          return ""
        }
        const episodeSeen = new Set<string>()
        const episodes = (r.episodes ?? [])
          .filter((ep) => {
            const key = toText(ep).trim()
            if (!key || episodeSeen.has(key)) return false
            episodeSeen.add(key)
            return true
          })
          .map((ep) => {
            const body = toText(ep)
            return {
              text: ep.valid_at ? `${body} (event_time: ${ep.valid_at})` : body,
              score: null as number | null,
              kind: "episode" as const,
            }
          })
        const communities = (r.communities ?? [])
          .filter((c) => c.name && c.summary)
          .map((c) => ({
            text: `${c.name}: ${c.summary}`,
            score: null as number | null,
            kind: "community" as const,
          }))
        return [...facts, ...entities, ...episodes, ...communities]
      }
      case "zep": {
        // Zep-paper dual scope baseline — mirrors locomo_search.py:
        //   asyncio.gather(
        //     graph.search(scope='nodes', reranker='rrf', limit=20),
        //     graph.search(scope='edges', reranker='cross_encoder', limit=20))
        // Then composes "entities (name: summary)" + "facts (fact, event_time: valid_at)".
        // We map to graphiti recipes:
        //   - NODE_HYBRID_SEARCH_RRF → nodes only, bm25+cosine, RRF
        //   - EDGE_HYBRID_SEARCH_CROSS_ENCODER → edges bm25+cosine+bfs, cross_encoder
        //     (BFS is in the recipe's search methods, but without bfs_origins it
        //     no-ops — smart=false means no cascade center resolution).
        const [nodeRes, edgeRes] = await Promise.all([
          client.kgDelve({
            bonfireId: config.bonfireId,
            query,
            numResults: 20,
            smart: false,
            searchRecipe: "NODE_HYBRID_SEARCH_RRF",
          }),
          client.kgDelve({
            bonfireId: config.bonfireId,
            query,
            numResults: 20,
            smart: false,
            searchRecipe: "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
          }),
        ])
        const entities = (nodeRes.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }))
        const facts = (edgeRes.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }))
        return [...facts, ...entities]
      }
      case "vector": {
        const r = await client.vectorSearch({
          bonfireId: config.bonfireId,
          query,
          limit: 10,
        })
        return r.map((h) => ({ text: h.text, score: h.score, kind: "chunk" as const }))
      }
      case "graph": {
        // Naive graph retrieval, but also consume entity summaries from
        // the :Entity side of /delve. Previously we only took edges
        // (flattenFacts), silently discarding the entities/episodes the
        // response also carries. 10 edges + 5 entity summaries = 15
        // items, no vector mixin.
        const r = await client.kgDelve({
          bonfireId: config.bonfireId,
          query,
          numResults: 10,
          smart: false,
        })
        const entities = (r.entities ?? [])
          .slice(0, 5)
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }))
        return [...flattenFacts(r).slice(0, 10), ...entities]
      }
      case "smart_graph": {
        // Smart kg_delve + entity summaries, but no vector mixin.
        // Isolates whether the entity-summary contribution is driving
        // the hybrid's win, or if vector chunks are still essential.
        // 15 edges + 5 entity summaries = 20 items.
        const r = await client.kgDelve({
          bonfireId: config.bonfireId,
          query,
          numResults: 15,
          smart: true,
        })
        const entities = (r.entities ?? [])
          .slice(0, 5)
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }))
        return [...flattenFacts(r).slice(0, 15), ...entities]
      }
      case "smart_naked": {
        // Zep published pattern without vectors: nodes RRF + edges
        // cross_encoder, 20 each = 40 items. Matches zep's LoCoMo eval
        // script verbatim (asyncio.gather of scope=nodes/rrf and
        // scope=edges/cross_encoder). Isolates whether the 5 vector chunks
        // in "smart" add signal on top of the paper-faithful zep baseline.
        const bfsScopesEnv = process.env.BONFIRES_BFS_SCOPES
        const rerankScopesEnv = process.env.BONFIRES_RERANK_SCOPES
        const bfsScopes = bfsScopesEnv
          ? (bfsScopesEnv
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges">)
          : undefined
        const rerankScopes = rerankScopesEnv
          ? (rerankScopesEnv
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges" | "episodes" | "communities">)
          : undefined
        const [nodeRes, edgeRes] = await Promise.all([
          client.kgDelve({
            bonfireId: config.bonfireId,
            query,
            numResults: 20,
            smart: true,
            searchRecipe: "NODE_HYBRID_SEARCH_RRF",
            bfsScopes,
            rerankScopes,
          }),
          client.kgDelve({
            bonfireId: config.bonfireId,
            query,
            numResults: 20,
            smart: true,
            searchRecipe: "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
            bfsScopes,
            rerankScopes,
          }),
        ])
        const entities = (nodeRes.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }))
        const facts = (edgeRes.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }))
        return [...facts, ...entities]
      }
      case "smart": {
        // Zep-pattern dual scope (nodes + edges in parallel) layered over
        // trimtab cascade enhancement. Each call goes through smart=true
        // so the cascade resolves a walk of entity UUIDs, enriches the
        // query with walk labels + leaf name, and seeds BFS from those
        // walk UUIDs at depth=1 (see delve search_service cascade path).
        //
        //   - Nodes: NODE_HYBRID_SEARCH_RRF — wide lexical/vector entity
        //     pool reranked by RRF; zep's choice for node scope.
        //   - Edges: EDGE_HYBRID_SEARCH_CROSS_ENCODER — bm25+cosine+bfs
        //     edge pool with cross-encoder rerank; zep's choice for fact
        //     precision.
        //
        // Both scopes benefit from the cascade: query enrichment widens
        // the text-matching surface, walk UUIDs as BFS origins anchor
        // expansion to the grammar-relevant neighborhood. 20+20 budget
        // matches zep's paper configuration.
        //
        // Benchmark knobs (env-driven, optional):
        //   BONFIRES_BFS_SCOPES=edges        → drops BFS from node_config
        //   BONFIRES_RERANK_SCOPES=nodes,edges → swaps non-listed scopes to RRF
        // Unset → default behavior (legacy).
        const bfsScopesEnv = process.env.BONFIRES_BFS_SCOPES
        const rerankScopesEnv = process.env.BONFIRES_RERANK_SCOPES
        const bfsScopes = bfsScopesEnv
          ? (bfsScopesEnv
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges">)
          : undefined
        const rerankScopes = rerankScopesEnv
          ? (rerankScopesEnv
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges" | "episodes" | "communities">)
          : undefined

        // Fire node+edge+vector concurrently. Vector adds per-message
        // recall that graphiti's entity/edge extraction may have missed
        // (property-style facts like "Caroline is single"). 5 chunks
        // is the empirical sweet spot on LoCoMo conv-26: v21 hit 60.3%
        // (v12 baseline = 58.29%) with 20 facts + 20 entities + 5 vec.
        // Mixing in episodes (v23) added noise on multi-hop and
        // hallucinations on adversarial — reverted.
        const [nodeRes, edgeRes, vectorRes] = await Promise.all([
          client.kgDelve({
            bonfireId: config.bonfireId,
            query,
            numResults: 20,
            smart: true,
            searchRecipe: "NODE_HYBRID_SEARCH_RRF",
            bfsScopes,
            rerankScopes,
          }),
          client.kgDelve({
            bonfireId: config.bonfireId,
            query,
            numResults: 20,
            smart: true,
            searchRecipe: "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
            bfsScopes,
            rerankScopes,
          }),
          client.vectorSearch({
            bonfireId: config.bonfireId,
            query,
            limit: 5,
          }),
        ])
        const entities = (nodeRes.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }))
        const facts = (edgeRes.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }))
        const chunks = (vectorRes ?? []).map((h) => ({
          text: h.text,
          score: h.score,
          kind: "chunk" as const,
        }))
        return [...facts, ...entities, ...chunks]
      }
      case "smart_hybrid": {
        // v32 — single round-trip to the unified /search/hybrid endpoint.
        // Server-side composes chunks_search + kgDelve(raw|enriched|fanout)
        // + hub_traversal in one shot, replacing the bench's previous
        // Promise.all over three calls (smart_unified). Same answer-LLM
        // input format ([CHUNK ...] / [ENTITY:Type] / [FACT]) so we can
        // A/B v30 (smart_unified) vs v32 (smart_hybrid) by env-toggling
        // BONFIRES_ARM with no other rendering changes.
        //
        // Env knobs (server owns mode/seed-gate, bench picks the mode):
        //   BONFIRES_HYBRID_MODE   — raw|enriched|enriched_gated|fanout
        //                            (default fanout: parallel raw+enriched)
        //   BONFIRES_HYBRID_RERANK — "0" disables the in-server point reranker
        //                            (default on; sub-ms cost vs ~500ms BGE)
        //   SMART_UNIFIED_*_LIMIT  — shared with smart_unified for parity
        const hybridMode = (process.env.BONFIRES_HYBRID_MODE ?? "fanout") as
          | "raw"
          | "enriched"
          | "enriched_gated"
          | "fanout"
        const useHybridRerank = process.env.BONFIRES_HYBRID_RERANK !== "0"
        const chunksLimit = parseInt(process.env.SMART_UNIFIED_CHUNKS_LIMIT ?? "20", 10)
        const entitiesLimit = parseInt(process.env.SMART_UNIFIED_ENTITIES_LIMIT ?? "10", 10)
        const factsLimit = parseInt(process.env.SMART_UNIFIED_FACTS_LIMIT ?? "20", 10)
        // The hybrid endpoint includes hub_facts by default; we let the
        // server default stand (include_hub_facts=true) and don't wire
        // BONFIRES_HUB_TRAVERSAL through — the legacy v30-era knob only
        // gated the bench-side hub call that smart_hybrid no longer makes.
        //
        // Server-side unified rerank (task #56): when BONFIRES_FINAL_RERANK=1
        // we ask delve to rerun a single CE pass over the merged pool of
        // chunks+entities+facts+hub_facts and surface the ranked top-N as
        // `unified_results`. Replaces the legacy bench-side ceRerank
        // round-trip — one fewer HTTP hop, identical CE model. Top-N
        // defaults to BONFIRES_FINAL_RERANK_TOP_N (matches the old gate).
        const useFinalRerank = process.env.BONFIRES_FINAL_RERANK === "1"
        const finalRerankTopN = parseInt(process.env.BONFIRES_FINAL_RERANK_TOP_N ?? "15", 10)
        // Two-pass enrichment knob (server-side flag). When BONFIRES_ENRICH_FROM_TOP_CHUNK=1
        // the delve server runs chunks_search(top_k=1) first, distills the top-1's
        // metadata (text/spacy_triples/taxonomy_labels/l2_label/timestamp) into an
        // enriched query, and uses that for the second-pass chunks + KG calls. The
        // top-1's kg_entity_uuid is also threaded into graphiti as an additional BFS
        // center seed. Costs +1 chunks_search round-trip; off by default for back-compat.
        const enrichFromTopChunk = process.env.BONFIRES_ENRICH_FROM_TOP_CHUNK === "1"
        // Parallel raw+enriched fanout (server-side flag). When
        // BONFIRES_ENRICHED_FANOUT=1 the delve server runs BOTH the raw and the
        // top-1-chunk-enriched query paths in parallel and unions the results
        // before the unified rerank. Combines raw's temporal/single-hop precision
        // with enriched's aggregate/multi-hop expansion. Implies enrich-from-top-chunk
        // server-side; ignored when pass-0 returns no chunks. Costs +1 chunks_search
        // round-trip + 2x kgDelve cost; off by default.
        const enrichedFanout = process.env.BONFIRES_ENRICHED_FANOUT === "1"
        // Confidence-gate threshold for enrichment (server-side). When >0,
        // delve only enriches if the pass-0 top-1 chunk has at least this
        // many distinct extracted entities in metadata.entities. Recommended
        // value: 3 — keeps enrichment on multi-hop questions (information-
        // rich top-1) and skips on vague single-hop pleasantries that dilute
        // the enriched query. 0 (default) = legacy always-enrich behavior.
        const enrichMinCenters = parseInt(process.env.BONFIRES_ENRICH_MIN_CENTERS ?? "0", 10)
        // Entity-lane positive gate: when set, only enrich if pass-0 top-1
        // chunk's aux_lane_breakdown contains the 'entity' lane (entity-name
        // token match fired). Empirically the cleanest discriminator on LoCoMo.
        const enrichRequireEntityLane = process.env.BONFIRES_ENRICH_REQUIRE_ENTITY_LANE === "1"
        // Community-cos inverted gate: when >0, skip enrichment if
        // aux_lane_breakdown['community_cos'] exceeds the threshold (vague
        // topical match → enrichment dilutes). Recommended: 0.008.
        const enrichMaxCommunityCos = parseFloat(
          process.env.BONFIRES_ENRICH_MAX_COMMUNITY_COS ?? "0"
        )
        // NLP-based question-shape gate. When set, the server classifies
        // the query and OVERRIDES the chunk-metadata gates above with a
        // "skip" / "enrich" / "neutral" verdict (mirrors hub_traversal's
        // detect_list_question philosophy).
        const enrichQuestionShapeGate = process.env.BONFIRES_ENRICH_QUESTION_SHAPE_GATE === "1"
        // Top-1 presearch (BONFIRES_ENRICH_CHUNKS_SEARCH=1): when the
        // enrichment gate passes, the server re-runs chunks_search with
        // the enriched query and uses the result as the final chunks
        // pool. Restores v40-enrich's chunks-side enrichment without the
        // fanout architecture. Adds ~3.5s when gate passes, 0s when blocked.
        const enrichChunksSearch = process.env.BONFIRES_ENRICH_CHUNKS_SEARCH === "1"
        // MMR diversification (BONFIRES_MMR_DIVERSIFY=1): re-orders the
        // unified rerank pool to balance relevance with diversity. Useful
        // for multi-aspect / list questions where CE rerank clusters
        // near-duplicate chunks at the top. lambda=1.0 → pure relevance
        // (no-op); 0.0 → pure diversity. Default 0.7.
        const mmrDiversify = process.env.BONFIRES_MMR_DIVERSIFY === "1"
        const mmrLambda = parseFloat(process.env.BONFIRES_MMR_LAMBDA ?? "0.7")
        // Per-lane MMR on the fact/edge list (BONFIRES_MMR_EDGES=1): runs
        // BEFORE the merged rerank pool is built, compressing 5+ near-
        // duplicate facts (e.g., "X pursues counseling" variants) into 1-2
        // representatives so diverse facts get top-N slots. Independent of
        // BONFIRES_MMR_DIVERSIFY; both can run together. Default lambda
        // 0.6 (slightly diversity-leaning) since fact near-duplication is
        // the worst across kinds.
        const mmrEdges = process.env.BONFIRES_MMR_EDGES === "1"
        const mmrEdgesLambda = parseFloat(process.env.BONFIRES_MMR_EDGES_LAMBDA ?? "0.6")
        // Disambiguated KG scopes (BONFIRES_DISAMBIGUATED_KG_SCOPES=1):
        // splits delve into entity-only + fact-only calls with scope-tuned
        // query formulations (nouns/categories for entities, verbs/temporal
        // for facts). Costs +1 delve call (~1-2s parallel) per arm; default
        // off for back-compat.
        const disambiguatedKgScopes = process.env.BONFIRES_DISAMBIGUATED_KG_SCOPES === "1"
        // Gate-time presearch target (BONFIRES_PRESEARCH_TARGET): which
        // trimtab symbol the top-1 enrichment lookup hits.
        //   "messages"   (default) — single conversation turns; matches
        //                 prior behavior.
        //   "aggregates" — cross-session preference/topic summaries
        //                 (PreferenceHub/TopicHub UUIDs); produces richer
        //                 enrichment seeds for broad/multi-aspect
        //                 questions.
        // No effect when enrichFromTopChunk is off.
        const presearchTarget = (process.env.BONFIRES_PRESEARCH_TARGET ?? "messages") as
          | "messages"
          | "aggregates"
        // Gate-time presearch source (BONFIRES_PRESEARCH_SOURCE): which
        // hydration source feeds the enrichment gate's top-1 metadata.
        //   "chunks"          (default) — chunks_search top-1 chunk drives
        //                     _build_top_chunk_enrichment (~3.5s, full
        //                     metadata blob).
        //   "trimtab_cascade" — chunks-grammar cascade walk emits a
        //                     resolved-refs string; wrapped as a single
        //                     synthetic top-1 (sub-second, token-light,
        //                     deterministic on grammar structure). The
        //                     chunks pool is then [synthetic_top1].
        // Orthogonal to BONFIRES_PRESEARCH_TARGET (which selects the
        // chunks-search SYMBOL — only used by the "chunks" source).
        const presearchSource = (process.env.BONFIRES_PRESEARCH_SOURCE ?? "chunks") as
          | "chunks"
          | "trimtab_cascade"
        // Cascade-first parallel pipeline (BONFIRES_CASCADE_FIRST_PIPELINE=1):
        // delve runs the trimtab cascade walk on the raw query first,
        // builds an enriched query from the walked text, and runs
        // chunks_search + kg_entity + kg_fact + hub in PARALLEL with the
        // enriched query (bypasses the pre-gate chunks_search + shape
        // gate logic). Requires the bonfire's primary_grammar to point
        // at a multi-symbol cascade grammar; falls back to legacy when
        // cascade returns empty. Independent of all other flags above.
        const cascadeFirstPipeline = process.env.BONFIRES_CASCADE_FIRST_PIPELINE === "1"
        // Corpus-aware pipeline routing mode. "legacy" (default)
        // preserves the cascadeFirstPipeline-driven dispatch. "auto"
        // invokes the server's QueryRouter per-query. "force_*" bypass
        // the router for A/B testing without restarting the server.
        const pipelineRoutingEnv = process.env.BONFIRES_PIPELINE_ROUTING
        const pipelineRouting:
          | "legacy"
          | "auto"
          | "force_chunks_first"
          | "force_cascade_first"
          | undefined =
          pipelineRoutingEnv === "auto" ||
          pipelineRoutingEnv === "force_chunks_first" ||
          pipelineRoutingEnv === "force_cascade_first" ||
          pipelineRoutingEnv === "legacy"
            ? pipelineRoutingEnv
            : undefined
        // Slim payload — strip fields the bench doesn't read so FastAPI
        // doesn't pay pydantic-validation + JSON-encode on multi-KB
        // metadata / debug. Default on; opt out via
        // BONFIRES_HYBRID_RESPONSE_LEAN=0 if a debug session needs the
        // full payload (elapsed_ms, gate state, chunk metadata blobs).
        const responseLean = process.env.BONFIRES_HYBRID_RESPONSE_LEAN !== "0"
        const smart = process.env.BONFIRES_HYBRID_SMART !== "0"
        const searchRecipeEnv = process.env.BONFIRES_SEARCH_RECIPE
        const includeHubWalk = process.env.BONFIRES_INCLUDE_HUB_WALK === "1" ? true : undefined
        const pickerOrChunkEnrich =
          process.env.BONFIRES_PICKER_OR_CHUNK_ENRICH === "1" ? true : undefined

        const res = await client.hybridSearch({
          bonfireId: config.bonfireId,
          query,
          topKChunks: chunksLimit,
          topKEntities: entitiesLimit,
          topKFacts: factsLimit,
          mode: hybridMode,
          smart,
          searchRecipe: searchRecipeEnv,
          rerank: useHybridRerank,
          rerankTopN: parseInt(
            process.env.BONFIRES_HYBRID_RERANK_TOP_N ?? String(Math.max(entitiesLimit, factsLimit)),
            10
          ),
          unifiedRerank: useFinalRerank,
          unifiedRerankTopN: finalRerankTopN,
          enrichFromTopChunk,
          enrichedFanout,
          enrichMinCenters,
          enrichRequireEntityLane,
          enrichMaxCommunityCos,
          enrichQuestionShapeGate,
          enrichChunksSearch,
          mmrDiversify,
          mmrLambda,
          mmrEdges,
          mmrEdgesLambda,
          disambiguatedKgScopes,
          presearchTarget,
          presearchSource,
          cascadeFirstPipeline,
          pipelineRouting,
          responseLean,
          includeHubWalk,
          pickerOrChunkEnrich,
          nowDate,
        })

        // When the server returned unified_results, prefer them — they
        // already encode the merged + CE-reranked top-N across all four
        // kinds. Skip the per-kind merge below entirely.
        if (res.unified_results && res.unified_results.length > 0) {
          return res.unified_results.map((item) => ({
            text: item.text,
            score: item.score,
            kind:
              item.kind === "hub_fact" || item.kind === "hub_walk"
                ? ("fact" as const)
                : (item.kind as "fact" | "claim" | "entity" | "chunk" | "answer_hint"),
          }))
        }

        // Drop PreferenceHub aggregates the same way smart_unified does.
        const dropHubEntities = process.env.BONFIRES_KEEP_HUB_ENTITIES !== "1"
        const entitiesU = (res.entities ?? [])
          .filter((e) => e.name && e.summary)
          .filter(
            (e) =>
              !dropHubEntities ||
              !(e.labels ?? []).some((lbl) => HYBRID_HUB_ENTITY_LABELS_TO_DROP.has(lbl))
          )
          .map((e) => {
            const specificLabel = (e.labels ?? []).find((lbl) => lbl && lbl !== "Entity")
            const tag = specificLabel ? `[ENTITY:${specificLabel}]` : "[ENTITY]"
            return {
              text: `${tag} ${e.name}: ${e.summary}`,
              score: null as number | null,
              kind: "entity" as const,
            }
          })

        const factsU = (res.edges ?? []).map((e) => ({
          text: e.valid_at ? `[FACT] ${e.fact} (event_time: ${e.valid_at})` : `[FACT] ${e.fact}`,
          score: e.score ?? null,
          kind: "fact" as const,
        }))

        // Chunk rendering — match smart_unified's [CHUNK date speaker] /
        // [PREFERENCE-SUMMARY ...] / [TOPIC-SUMMARY ...] tag conventions
        // so the answer LLM sees identical input format across both arms.
        const chunksU = (res.chunks ?? []).map((h) => {
          const meta = (h.metadata ?? {}) as {
            speaker?: string | null
            timestamp?: string | null
            type?: string | null
          }
          const date = meta.timestamp ? String(meta.timestamp).slice(0, 10) : ""
          const speaker = meta.speaker ?? ""
          const chunkType = meta.type ?? ""
          let tag: string
          if (chunkType === "preference") {
            tag =
              date && speaker
                ? `[PREFERENCE-SUMMARY ${date} ${speaker}]`
                : speaker
                  ? `[PREFERENCE-SUMMARY ${speaker}]`
                  : "[PREFERENCE-SUMMARY]"
          } else if (chunkType === "topic") {
            tag =
              date && speaker
                ? `[TOPIC-SUMMARY ${date} ${speaker}]`
                : speaker
                  ? `[TOPIC-SUMMARY ${speaker}]`
                  : "[TOPIC-SUMMARY]"
          } else if (date && speaker) {
            tag = `[CHUNK ${date} ${speaker}]`
          } else if (speaker) {
            tag = `[CHUNK ${speaker}]`
          } else if (date) {
            tag = `[CHUNK ${date}]`
          } else {
            tag = "[CHUNK]"
          }
          const cleanText = h.text.startsWith("[") ? h.text.replace(/^\[[^\]]*\]\s*/, "") : h.text
          return {
            text: `${tag} ${cleanText}`,
            score: h.score,
            kind: "chunk" as const,
          }
        })

        // Hub-traversal facts merge into the [FACT] bucket — same as smart_unified.
        const hubFactsU = (res.hub_facts ?? []).map((f) => ({
          text: f.text.startsWith("[") ? f.text : `[FACT] ${f.text}`,
          score: f.score,
          kind: "fact" as const,
        }))

        // Fall-through path: server didn't return unified_results (either
        // BONFIRES_FINAL_RERANK=0, or the server build predates task #56).
        // Use the per-kind arrays as before. The legacy bench-side ceRerank
        // round-trip is gone — when the user wants final-stage rerank they
        // get it server-side via unified_rerank (no extra HTTP hop).
        const merged = [...factsU, ...entitiesU, ...chunksU, ...hubFactsU]
        return merged
      }
      case "smart_unified":
      case "smart_cascade": {
        // smart_unified: zep dual-scope graph + N chunks in parallel, chunks
        // via trimtab HybridRetriever + cross-encoder rerank.
        // smart_cascade: same, but chunks request the HyperMem-lite Phase A
        // gate (top-k taxonomy labels narrow the candidate pool before
        // dense/BM25/rerank). v27i1 — tests whether hierarchical
        // restriction lifts multi-hop without regressing single-hop.
        const useCascade = config.arm === "smart_cascade"
        const bfsScopesEnvU = process.env.BONFIRES_BFS_SCOPES
        const rerankScopesEnvU = process.env.BONFIRES_RERANK_SCOPES
        const bfsScopesU = bfsScopesEnvU
          ? (bfsScopesEnvU
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges">)
          : undefined
        const rerankScopesU = rerankScopesEnvU
          ? (rerankScopesEnvU
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges" | "episodes" | "communities">)
          : undefined
        // Arm quotas: defaults are v26i7 (20 chunks + 10 entities + 20 edges).
        // v28i3 inverts the chunk/entity trade: rank analysis showed only
        // 3.9% of correct answers came from rank 5+ chunks, so we trade
        // chunk depth (20→10) for KG density (entities 10→20). Facts stay
        // at 20 because multi-hop hits rank 1 from facts (26/27 correct).
        // Env overrides for A/B.
        const chunksLimit = parseInt(process.env.SMART_UNIFIED_CHUNKS_LIMIT ?? "20", 10)
        const entitiesLimit = parseInt(process.env.SMART_UNIFIED_ENTITIES_LIMIT ?? "10", 10)
        const factsLimit = parseInt(process.env.SMART_UNIFIED_FACTS_LIMIT ?? "20", 10)
        // Hub-traversal lane (BONFIRES_HUB_TRAVERSAL=1). Default off so
        // existing benches stay byte-equivalent. When enabled, runs in
        // parallel with the three existing lanes; failure is non-fatal.
        const hubTraversalEnabled = process.env.BONFIRES_HUB_TRAVERSAL === "1"
        // Phase 1a probe: when BONFIRES_KG_QUERY_FROM_CHUNK=1, run chunksSearch
        // FIRST and use its top-1 chunk text as the query for the kgDelve calls
        // (cascade walk + BM25/cosine over chunk-grounded text instead of the
        // raw question). HyDE-like seeding test. Falls back to raw query when
        // chunksSearch returns nothing. Adds chunks_search wall-clock to the
        // critical path, but kgDelve is the slow lane so net effect is small.
        const useChunkAsKgQuery = process.env.BONFIRES_KG_QUERY_FROM_CHUNK === "1"
        // Fan-out mode: run kgDelve TWICE — once with the raw query, once with
        // the chunk-seeded enriched query — and merge their entities/edges
        // (dedupe by uuid/fact) before passing to the answer pool. Spot tests
        // showed raw wins narrow-intent (q3 "Caroline research") and enriched
        // wins broad/list (q19 "Melanie's kids likes"); the union catches both.
        // Mutually exclusive with useChunkAsKgQuery (the legacy enriched-only path).
        const kgseedFanout = process.env.BONFIRES_KGSEED_FANOUT === "1"
        // Enriched-only mode (v31): use enriched query for the single kgDelve,
        // but cheap-protect against off-topic seeds via entity-overlap gating.
        // If the seed chunk's metadata.entities (GLiNER spans) have ZERO overlap
        // with the question's noun tokens, the seed is high-cosine but off-topic
        // (q3 "Caroline research" → seed was Melanie/pottery) — fall back to raw
        // query. Pure metadata check, no API call. Mutex with fan-out.
        const kgseedOnly = process.env.BONFIRES_KGSEED_ONLY === "1"
        let chunkHits: Awaited<ReturnType<typeof client.chunksSearch>> = []
        let kgQuery = query
        if (useChunkAsKgQuery || kgseedFanout || kgseedOnly) {
          chunkHits = await client.chunksSearch({
            bonfireId: config.bonfireId,
            query,
            limit: chunksLimit,
            nowDate,
            cascade: useCascade,
          })
          // Pick the first NON-AGGREGATE chunk for seeding. Preference /
          // topic aggregates score high on broad queries (they union many
          // signals per speaker) but they over-route the kg lane toward
          // their cluster. Specific raw turn chunks anchor the cascade
          // walk and BM25 surface to the actual answer-bearing event.
          const isAggregate = (h: (typeof chunkHits)[number]): boolean => {
            const t = (h.metadata as { type?: string | null } | null)?.type
            return t === "preference" || t === "topic"
          }
          const seedChunk = chunkHits.find((h) => !isAggregate(h)) ?? chunkHits[0]
          // Entity-overlap gate: when kgseedOnly is on, refuse the seed if its
          // GLiNER entity spans share zero tokens with the question (case-insensitive,
          // length>3 to skip short stopwords). Falls back to raw query — preserves
          // intent for narrow-question failures of the seed picker.
          let acceptSeed = true
          if (kgseedOnly && seedChunk) {
            const STOP = new Set([
              "the",
              "and",
              "for",
              "with",
              "what",
              "does",
              "what's",
              "that",
              "this",
              "when",
              "were",
              "have",
              "has",
              "had",
              "will",
              "into",
              "from",
              "about",
              "whom",
              "there",
              "their",
              "they",
              "them",
            ])
            const qTokens = new Set(
              query
                .toLowerCase()
                .split(/[^a-z0-9]+/)
                .filter((t) => t.length > 3 && !STOP.has(t))
            )
            const meta = (seedChunk.metadata ?? {}) as {
              entities?: Record<string, string[]> | null
            }
            const seedTokens = new Set<string>()
            for (const spans of Object.values(meta.entities ?? {})) {
              if (!Array.isArray(spans)) continue
              for (const s of spans) {
                if (typeof s !== "string") continue
                for (const t of s.toLowerCase().split(/[^a-z0-9]+/)) {
                  if (t.length > 3) seedTokens.add(t)
                }
              }
            }
            const overlap = [...qTokens].filter((t) => seedTokens.has(t))
            acceptSeed = overlap.length > 0
            if (!acceptSeed) {
              // Off-topic seed — leave kgQuery=query so kgDelve uses raw intent.
              // Skip the enrichment block below.
            }
          }
          if (seedChunk && acceptSeed) {
            // Build a DISTILLED enriched query: original query leads (so BM25
            // anchors on user intent), followed by the seed chunk's already-
            // extracted metadata (atomic_claims, GLiNER entity spans, l2
            // label, speaker/date). The pre-extracted forms are far cleaner
            // BM25 tokens than 400 chars of conversational prose — they
            // carry the same discriminative signal at ~10x token efficiency.
            // Stays well under the embedder's 512-token limit.
            //
            // Replaces the earlier "chunk text leads + original trails"
            // format that buried single-hop intent under chunk-specific
            // tokens (regressed single-hop 50% → 28% on the 199q bench).
            const meta = (seedChunk.metadata ?? {}) as {
              speaker?: string | null
              timestamp?: string | null
              type?: string | null
              l2_label?: string | null
              entities?: Record<string, string[]> | null
              atomic_claims?: Array<{ text?: string }> | null
            }
            const parts: string[] = [query]

            const speaker = meta.speaker ?? ""
            const date = meta.timestamp ? String(meta.timestamp).slice(0, 10) : ""
            if (speaker || date) {
              parts.push(`@ ${[speaker, date].filter(Boolean).join(" ")}`)
            }

            // Atomic claims — pre-extracted single-proposition spans. For a
            // chunk that says "we went hiking, painting, and camping" these
            // come back as 3 separate one-line claims. Top-5 keeps it lean.
            const claims = (meta.atomic_claims ?? [])
              .map((c) => (c?.text ?? "").trim())
              .filter((t) => t.length > 0)
              .slice(0, 5)
            if (claims.length) parts.push(`claims: ${claims.join("; ")}`)

            // GLiNER-extracted entity spans (proper nouns + typed mentions).
            // Flatten {Person: [...], Activity: [...]} into a name list.
            const entSpans: string[] = []
            const ents = meta.entities ?? {}
            for (const spans of Object.values(ents)) {
              if (Array.isArray(spans)) {
                for (const s of spans) {
                  if (typeof s === "string" && s.trim()) entSpans.push(s.trim())
                }
              }
            }
            const uniqEnts = Array.from(new Set(entSpans)).slice(0, 8)
            if (uniqEnts.length) parts.push(`mentions: ${uniqEnts.join(", ")}`)

            // L2 ontology category — single word like "Family_Member" / "Activity"
            if (meta.l2_label) parts.push(`type: ${meta.l2_label}`)

            kgQuery = parts.join(" | ")
          }
        }
        // Single combined kgDelve replaces the previous NODE + EDGE pair —
        // halves cascade-walk cost (the smart=True cascade resolution runs
        // ONCE per request now instead of twice). COMBINED_HYBRID_SEARCH_*
        // recipes return entities AND edges from a single Graphiti call;
        // bench truncates to per-kind limits client-side.
        //   - bfsScopes=["edges"] explicitly: Edge BFS is bounded; Node BFS
        //     explodes on dense per-message graphs (delve docstring measures
        //     ~36s at depth=3). Lock it off regardless of env.
        //   - searchRecipe: COMBINED_RRF when final-rerank is on (cheap, the
        //     merge step does the BGE pass); COMBINED_CROSS_ENCODER otherwise
        //     so per-scope Voyage CE produces the final ordering.
        // COMBINED_HYBRID_SEARCH_RRF unconditionally — the _CROSS_ENCODER
        // variant currently 500s on graphiti's EntityEdge pydantic validation
        // (episodes=None vs required list). RRF returns clean dual-scope
        // results in ~2s. Rely on chunks-side rerank (or BONFIRES_FINAL_RERANK
        // merge-side if enabled) for final ordering rather than per-scope CE.
        const combinedRecipe = "COMBINED_HYBRID_SEARCH_RRF"
        const combinedLimit = Math.max(entitiesLimit, factsLimit)
        // Fan-out: parallel raw + enriched kgDelve. The shared chunksSearch
        // result also lands here when fanout is on (already awaited above).
        const kgRawPromise = client.kgDelve({
          bonfireId: config.bonfireId,
          query, // ALWAYS raw — captures user intent, fixes single-hop tank
          numResults: combinedLimit,
          smart: true,
          searchRecipe: combinedRecipe,
          bfsScopes: ["edges"],
          rerankScopes: rerankScopesU,
        })
        const kgEnrichedPromise =
          (kgseedFanout || useChunkAsKgQuery || kgseedOnly) && kgQuery !== query
            ? client.kgDelve({
                bonfireId: config.bonfireId,
                query: kgQuery, // chunk-seeded enriched query
                numResults: combinedLimit,
                smart: true,
                searchRecipe: combinedRecipe,
                bfsScopes: ["edges"],
                rerankScopes: rerankScopesU,
              })
            : null

        // Mode matrix:
        //   default            → raw only
        //   useChunkAsKgQuery  → enriched only (legacy, no gate)
        //   kgseedOnly         → enriched if seed accepted, else raw — single call
        //   kgseedFanout       → both raw and enriched in parallel, merged
        const skipRawCall =
          (useChunkAsKgQuery && !kgseedFanout) || (kgseedOnly && kgQuery !== query)
        const [combinedRawRes, combinedEnrichedRes, chunkHitsParallel, hubFactsRes] =
          await Promise.all([
            skipRawCall
              ? Promise.resolve({ entities: [], edges: [] } as Awaited<typeof kgRawPromise>)
              : kgRawPromise,
            kgEnrichedPromise ??
              Promise.resolve({ entities: [], edges: [] } as Awaited<typeof kgRawPromise>),
            useChunkAsKgQuery || kgseedFanout || kgseedOnly
              ? Promise.resolve(chunkHits)
              : client.chunksSearch({
                  bonfireId: config.bonfireId,
                  query,
                  limit: chunksLimit,
                  nowDate,
                  cascade: useCascade,
                }),
            hubTraversalEnabled
              ? client
                  .hubTraversal({ bonfireId: config.bonfireId, query })
                  .catch((err: unknown) => {
                    console.warn(`hubTraversal failed (non-fatal): ${err}`)
                    return {
                      facts: [] as Array<{ text: string; kind: string; score: number | null }>,
                    }
                  })
              : Promise.resolve({
                  facts: [] as Array<{ text: string; kind: string; score: number | null }>,
                }),
          ])

        // Merge raw + enriched: union by uuid (entities) and fact text (edges).
        // Order: raw first, enriched appended (raw wins ties on intent-match).
        // The merge-side BGE rerank (BONFIRES_FINAL_RERANK=1) reranks across
        // the union, so per-side rank order matters less than CE quality.
        const seenEntityUuids = new Set<string>()
        const mergedEntities: NonNullable<typeof combinedRawRes.entities> = []
        for (const src of [combinedRawRes, combinedEnrichedRes]) {
          for (const e of src.entities ?? []) {
            const key = e.uuid || e.name || ""
            if (!key || seenEntityUuids.has(key)) continue
            seenEntityUuids.add(key)
            mergedEntities.push(e)
          }
        }
        const seenFacts = new Set<string>()
        const mergedEdges: NonNullable<typeof combinedRawRes.edges> = []
        for (const src of [combinedRawRes, combinedEnrichedRes]) {
          for (const ed of src.edges ?? []) {
            const key = ed.fact || ""
            if (!key || seenFacts.has(key)) continue
            seenFacts.add(key)
            mergedEdges.push(ed)
          }
        }
        // After merging, the per-kind slices use the doubled limit so the
        // CE rerank gets the full union to choose from. When fan-out is off
        // mergedEntities == raw entities (other side empty), so behavior
        // matches the pre-fanout path byte-for-byte.
        const slicedEntityLimit =
          kgseedFanout && combinedEnrichedRes.entities?.length ? entitiesLimit * 2 : entitiesLimit
        const slicedFactLimit =
          kgseedFanout && combinedEnrichedRes.edges?.length ? factsLimit * 2 : factsLimit
        const nodeResU = { entities: mergedEntities.slice(0, slicedEntityLimit) }
        const edgeResU = { edges: mergedEdges.slice(0, slicedFactLimit) }
        chunkHits = chunkHitsParallel
        void bfsScopesU // currently locked to ["edges"]; env override deferred
        // Enrich entity hits with their specific taxonomy label when the
        // KG carries one beyond the generic ``:Entity`` base — the answer
        // LLM can then tell a ``:Pet`` from a ``:Book_Title`` instead of
        // seeing name+summary alone.
        //
        // Drop graphiti's per-speaker/per-session aggregate-hub entities
        // from the rerank pool. They have heavy subject+topic mention so
        // BGE consistently ranks them top-1 for World-Knowledge questions,
        // burying the actual answer-bearing facts. Their member chunks
        // are already in the chunks lane (and aggregate-propagation
        // surfaces them via the cascade); we don't need the aggregate
        // entity itself competing in rerank. Default-on; opt out via
        // BONFIRES_KEEP_HUB_ENTITIES=1 for A/B.
        // Only drop PreferenceHub — the per-speaker aggregate that unions ALL
        // preferences for that speaker, which gives it max BGE cosine to any
        // question mentioning the speaker + any topic. PreferenceRecord and
        // TopicRecord are per-session content-rich aggregates that carry
        // useful soft-context (specific items, events, places) — keep them.
        const HUB_ENTITY_LABELS_TO_DROP = new Set(["PreferenceHub"])
        const dropHubEntities = process.env.BONFIRES_KEEP_HUB_ENTITIES !== "1"
        // Uniform kind tagging in the rendered text so the answer LLM
        // can reason about provenance directly when no cross-encoder
        // rerank fires (BONFIRES_FINAL_RERANK=0). The merged-pool order
        // becomes the LLM's job to filter; explicit `[ENTITY:Pet]`,
        // `[FACT]`, `[CHUNK ...]`, `[PREFERENCE-SUMMARY ...]`,
        // `[TOPIC-SUMMARY ...]` tags let it weight by kind.
        const entitiesU = (nodeResU.entities ?? [])
          .filter((e) => e.name && e.summary)
          .filter(
            (e) =>
              !dropHubEntities ||
              !(e.labels ?? []).some((lbl) => HUB_ENTITY_LABELS_TO_DROP.has(lbl))
          )
          .map((e) => {
            const specificLabel = (e.labels ?? []).find((lbl) => lbl && lbl !== "Entity")
            const tag = specificLabel ? `[ENTITY:${specificLabel}]` : "[ENTITY]"
            return {
              text: `${tag} ${e.name}: ${e.summary}`,
              score: null as number | null,
              kind: "entity" as const,
            }
          })
        const factsU = (edgeResU.edges ?? []).map((e) => ({
          text: e.valid_at ? `[FACT] ${e.fact} (event_time: ${e.valid_at})` : `[FACT] ${e.fact}`,
          score: e.score ?? null,
          kind: "fact" as const,
        }))
        // Chunks carry speaker + timestamp + type. Aggregates (preference
        // / topic summaries) live in the ``aggregates`` grammar symbol
        // server-side and never come back from chunks_search — they reach
        // top-K only through the aggregate→member RRF boost lane (which
        // surfaces their member chunks) and through PreferenceRecord /
        // TopicRecord :Entity nodes in kgDelve.
        const chunksU = chunkHits.map((h) => {
          const meta = (h.metadata ?? {}) as {
            speaker?: string | null
            timestamp?: string | null
            type?: string | null
          }
          const date = meta.timestamp ? String(meta.timestamp).slice(0, 10) : ""
          const speaker = meta.speaker ?? ""
          const chunkType = meta.type ?? ""
          let tag: string
          if (chunkType === "preference") {
            tag =
              date && speaker
                ? `[PREFERENCE-SUMMARY ${date} ${speaker}]`
                : speaker
                  ? `[PREFERENCE-SUMMARY ${speaker}]`
                  : "[PREFERENCE-SUMMARY]"
          } else if (chunkType === "topic") {
            tag =
              date && speaker
                ? `[TOPIC-SUMMARY ${date} ${speaker}]`
                : speaker
                  ? `[TOPIC-SUMMARY ${speaker}]`
                  : "[TOPIC-SUMMARY]"
          } else if (date && speaker) {
            tag = `[CHUNK ${date} ${speaker}]`
          } else if (speaker) {
            tag = `[CHUNK ${speaker}]`
          } else if (date) {
            tag = `[CHUNK ${date}]`
          } else {
            tag = "[CHUNK]"
          }
          // Strip any pre-existing `[...]` prefix so we don't double-tag.
          const cleanText = h.text.startsWith("[") ? h.text.replace(/^\[[^\]]*\]\s*/, "") : h.text
          return {
            text: `${tag} ${cleanText}`,
            score: h.score,
            kind: "chunk" as const,
          }
        })
        const hubFactsU = (hubFactsRes.facts ?? []).map(
          (f: { text: string; score: number | null }) => ({
            // Hub-traversal facts merge into the [FACT] bucket — the LLM
            // doesn't need to distinguish their provenance, just consume them
            // alongside graphiti edge facts.
            text: f.text.startsWith("[") ? f.text : `[FACT] ${f.text}`,
            score: f.score,
            kind: "fact" as const,
          })
        )
        const merged = [...factsU, ...entitiesU, ...chunksU, ...hubFactsU]

        // Note: the legacy BONFIRES_FINAL_RERANK=1 ceRerank round-trip
        // has been removed (task #56). For server-side unified rerank
        // across all kinds use the smart_hybrid arm — it now requests
        // `unified_rerank` from delve directly and avoids the extra
        // HTTP hop. The smart_unified/smart_cascade arms compose their
        // pool client-side (parallel kgDelve + chunksSearch + hubTraversal)
        // so a unified-pool CE rerank doesn't have a server-side hook
        // here; the merged order stands.
        return merged
      }
    }
  } catch (err) {
    console.error(`bonfires search (${config.arm}) failed:`, err)
    return []
  }
}
