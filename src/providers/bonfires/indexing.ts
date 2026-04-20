import type { BonfiresClient } from "./client.js";

/**
 * Run the full post-ingest indexing pipeline for a bonfire.
 *
 * Order:
 *   startSummaries  → waitForJob
 *   startTaxonomy   → waitForJob
 *   buildCommunities
 *   buildGrammar    (POST /trimtabs/grammars/{id}/build — the LangGraph
 *                   GrammarBuilderGraphCoordinator self-heals taxonomy
 *                   uuids, creating a :Taxonomy KG entity per orphan and
 *                   saving the uuid back to Mongo. See delve's
 *                   grammar_builder_graph_coordinator._fetch_taxonomy_labels.)
 *
 * stack_process runs per-session inside ingestSessions; not repeated here.
 */
export async function runIndexingPipeline(args: {
  client: BonfiresClient;
  agentId: string;
  bonfireId: string;
}): Promise<void> {
  const { client, bonfireId } = args;

  const summaries = await client.startSummaries(bonfireId);
  await client.waitForJob(summaries.job_id, { kind: "summaries", timeoutSec: 1800 });

  const taxonomy = await client.startTaxonomy(bonfireId);
  await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 });

  await client.buildCommunities(bonfireId);

  await client.buildGrammar(bonfireId);
}
