import type { BonfiresClient } from "./client.js";
import type { BonfiresConfig, KgDelveResult } from "./types.js";

export interface SearchHit {
  text: string;
  score: number | null;
}

export function flattenFacts(result: KgDelveResult): SearchHit[] {
  const edges = result.edges ?? [];
  return edges.map((e) => ({ text: e.fact, score: e.score ?? null }));
}

export async function armSearch(args: {
  client: Pick<BonfiresClient, "vectorSearch" | "kgDelve">;
  query: string;
  config: BonfiresConfig;
}): Promise<SearchHit[]> {
  const { client, query, config } = args;
  try {
    switch (config.arm) {
      case "vector": {
        const r = await client.vectorSearch({
          bonfireId: config.bonfireId,
          query,
          limit: 5,
        });
        return r.map((h) => ({ text: h.text, score: h.score }));
      }
      case "graph": {
        const r = await client.kgDelve({
          bonfireId: config.bonfireId,
          query,
          numResults: 5,
          smart: false,
          autoResolveCenter: false,
        });
        return flattenFacts(r).slice(0, 5);
      }
      case "smart": {
        const r = await client.kgDelve({
          bonfireId: config.bonfireId,
          query,
          numResults: 5,
          smart: true,
        });
        return flattenFacts(r).slice(0, 5);
      }
    }
  } catch (err) {
    console.error(`bonfires search (${config.arm}) failed:`, err);
    return [];
  }
}
