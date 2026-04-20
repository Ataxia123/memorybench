import type { BonfiresClient } from "./client.js";

/**
 * Run the full post-ingest indexing pipeline for a bonfire.
 *
 * Order (matches whodunit):
 *   startSummaries → waitForJob
 *   startTaxonomy  → waitForJob
 *   buildCommunities
 *   buildOntology
 *   createGrammar + seedGrammar
 *
 * NOTE: stack_process is intentionally absent here — ingestSessions already
 * calls it per session so each session becomes its own KG episode.
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

  // 1. Summaries over ingested chunks
  const summaries = await client.startSummaries(bonfireId);
  await client.waitForJob(summaries.job_id, { kind: "summaries", timeoutSec: 1800 });

  // 2. Taxonomy
  const taxonomy = await client.startTaxonomy(bonfireId);
  await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 });

  // 3. Communities (prerequisite for ontology)
  await client.buildCommunities(bonfireId);

  // 4. Ontology — wires :OntologyLabel nodes + :BELONGS_TO edges
  await client.buildOntology(bonfireId, { linkToGraph: true });

  // 5. Grammar: create + seed (not build_grammar — fresh bonfires don't have
  //    KG-UUID-linked taxonomies, so we seed from a KG query instead)
  await client.createGrammar({ bonfireId, grammar: grammarName });
  await client.seedGrammar({
    bonfireId,
    grammar: grammarName,
    rule: "entities",
    kgQuery: seedQuery,
    numEntities: 30,
  });
}
