import type { BonfiresClient } from "./client.js";

/**
 * Run the full post-ingest indexing pipeline for a bonfire.
 *
 * Order (applicant-review-aligned):
 *   startSummaries  → waitForJob
 *   startTaxonomy   → waitForJob
 *   startLabeling   → waitForJob       # assigns KG UUIDs to taxonomy labels
 *   buildCommunities
 *   buildGrammar                       # POST /trimtabs/grammars/{id}/build
 *
 * `buildGrammar` runs Delve's `GrammarBuilderGraphCoordinator`, which auto-
 * reads taxonomies + ontology (falling back to ["world"] when the bonfire
 * has no Ontology doc). `/labeling/hybrid` is required so taxonomies carry
 * the KG UUID the coordinator needs to populate expansions.
 *
 * stack_process is absent here — ingestSessions calls it per session so
 * each session becomes its own KG episode.
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

  const labeling = await client.startLabeling(bonfireId);
  await client.waitForJob(labeling.job_id, { kind: "labeling", timeoutSec: 1800 });

  await client.buildCommunities(bonfireId);

  await client.buildGrammar(bonfireId);
}
