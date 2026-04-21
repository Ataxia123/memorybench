import type { BonfiresClient } from "./client.js";
import type { BonfiresConfig, KgDelveResult } from "./types.js";

export interface SearchHit {
  text: string;
  score: number | null;
  /** Scope tag lets the answer prompt categorize the hit into zep-style
   *  <FACTS> vs <ENTITIES> sections. "fact" = edges (relationship claims
   *  with valid_at), "entity" = nodes (name + summary). Optional — hits
   *  without a kind fall through as "other" (treated as facts). */
  kind?: "fact" | "entity" | "episode" | "community" | "chunk";
}

export function flattenFacts(result: KgDelveResult): SearchHit[] {
  const edges = result.edges ?? [];
  return edges.map((e) => ({ text: e.fact, score: e.score ?? null, kind: "fact" as const }));
}

export async function armSearch(args: {
  client: Pick<BonfiresClient, "vectorSearch" | "kgDelve">;
  query: string;
  config: BonfiresConfig;
}): Promise<SearchHit[]> {
  const { client, query, config } = args;
  try {
    switch (config.arm) {
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
        const bfsScopesEnvFull = process.env.BONFIRES_BFS_SCOPES;
        const rerankScopesEnvFull = process.env.BONFIRES_RERANK_SCOPES;
        const bfsScopesFull = bfsScopesEnvFull
          ? (bfsScopesEnvFull.split(",").map((s) => s.trim()).filter(Boolean) as Array<"nodes" | "edges">)
          : undefined;
        const rerankScopesFull = rerankScopesEnvFull
          ? (rerankScopesEnvFull
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges" | "episodes" | "communities">)
          : undefined;
        const r = await client.kgDelve({
          bonfireId: config.bonfireId,
          query,
          numResults: 20,
          smart: true,
          bfsScopes: bfsScopesFull,
          rerankScopes: rerankScopesFull,
        });
        const entities = (r.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }));
        const facts = (r.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }));
        const episodeSeen = new Set<string>();
        const episodes = (r.episodes ?? [])
          .filter((ep) => {
            const key = (ep.content ?? ep.summary ?? "").trim();
            if (!key || episodeSeen.has(key)) return false;
            episodeSeen.add(key);
            return true;
          })
          .map((ep) => ({
            text: ep.valid_at
              ? `${ep.content ?? ep.summary ?? ""} (event_time: ${ep.valid_at})`
              : (ep.content ?? ep.summary ?? ""),
            score: null as number | null,
            kind: "episode" as const,
          }));
        const communities = (r.communities ?? [])
          .filter((c) => c.name && c.summary)
          .map((c) => ({
            text: `${c.name}: ${c.summary}`,
            score: null as number | null,
            kind: "community" as const,
          }));
        return [...facts, ...entities, ...episodes, ...communities];
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
        ]);
        const entities = (nodeRes.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }));
        const facts = (edgeRes.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }));
        return [...facts, ...entities];
      }
      case "vector": {
        const r = await client.vectorSearch({
          bonfireId: config.bonfireId,
          query,
          limit: 10,
        });
        return r.map((h) => ({ text: h.text, score: h.score, kind: "chunk" as const }));
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
        });
        const entities = (r.entities ?? [])
          .slice(0, 5)
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }));
        return [...flattenFacts(r).slice(0, 10), ...entities];
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
        });
        const entities = (r.entities ?? [])
          .slice(0, 5)
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }));
        return [...flattenFacts(r).slice(0, 15), ...entities];
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
        const bfsScopesEnv = process.env.BONFIRES_BFS_SCOPES;
        const rerankScopesEnv = process.env.BONFIRES_RERANK_SCOPES;
        const bfsScopes = bfsScopesEnv
          ? (bfsScopesEnv.split(",").map((s) => s.trim()).filter(Boolean) as Array<"nodes" | "edges">)
          : undefined;
        const rerankScopes = rerankScopesEnv
          ? (rerankScopesEnv
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean) as Array<"nodes" | "edges" | "episodes" | "communities">)
          : undefined;

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
        ]);
        const entities = (nodeRes.entities ?? [])
          .filter((e) => e.name && e.summary)
          .map((e) => ({
            text: `${e.name}: ${e.summary}`,
            score: null as number | null,
            kind: "entity" as const,
          }));
        const facts = (edgeRes.edges ?? []).map((e) => ({
          text: e.valid_at ? `${e.fact} (event_time: ${e.valid_at})` : e.fact,
          score: e.score ?? null,
          kind: "fact" as const,
        }));
        return [...facts, ...entities];
      }
    }
  } catch (err) {
    console.error(`bonfires search (${config.arm}) failed:`, err);
    return [];
  }
}
