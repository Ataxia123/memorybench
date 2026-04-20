import type { BonfiresClient } from "./client.js";

/**
 * Run the full post-ingest indexing pipeline for a bonfire.
 *
 * Order (matches whodunit's pipeline.py):
 *   startSummaries → waitForJob
 *   startTaxonomy  → waitForJob
 *   buildCommunities
 *   createGrammar + seedGrammar
 *
 * Ontology is intentionally omitted — Delve's `POST /ontology/{id}/build`
 * needs hand-authored labels (not derivable from taxonomies) and whodunit's
 * production pipeline doesn't include it. Taxonomies + communities +
 * KG-seeded grammar is sufficient for the smart arm's cascade.
 *
 * stack_process is also absent here — ingestSessions calls it per session
 * so each session becomes its own KG episode.
 */
export async function runIndexingPipeline(args: {
  client: BonfiresClient;
  agentId: string;
  bonfireId: string;
  grammarName?: string;
  seedQuery?: string;
}): Promise<void> {
  const { client, bonfireId } = args;
  const grammarName = args.grammarName ?? "locomo";
  const seedQuery = args.seedQuery ?? "people places events topics";

  const summaries = await client.startSummaries(bonfireId);
  await client.waitForJob(summaries.job_id, { kind: "summaries", timeoutSec: 1800 });

  const taxonomy = await client.startTaxonomy(bonfireId);
  await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 });

  await client.buildCommunities(bonfireId);

  await client.createGrammar({ bonfireId, grammar: grammarName });
  await client.seedGrammar({
    bonfireId,
    grammar: grammarName,
    rule: "entities",
    kgQuery: seedQuery,
    numEntities: 30,
  });
}
