import type { BonfiresClient } from "./client.js";

export async function runIndexingPipeline(args: {
  client: BonfiresClient;
  agentId: string;
  bonfireId: string;
}): Promise<void> {
  const { client, agentId, bonfireId } = args;

  // 1. Build the KG from the agent's stack.
  const { task_id } = await client.stackProcess(agentId);
  await client.waitForJob(task_id, { kind: "stack_processing", timeoutSec: 1800 });

  // 2. Cluster entities into taxonomies.
  const taxonomy = await client.startTaxonomy(bonfireId);
  await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 });

  // 3. Detect communities (Leiden) — must run BEFORE ontology build.
  await client.buildCommunities(bonfireId);

  // 4. Build ontology — wires :OntologyLabel nodes + :BELONGS_TO edges.
  //    Requires taxonomies + communities to exist.
  await client.buildOntology(bonfireId, { linkToGraph: true });

  // 5. Build the trimtab grammar last — cascade needs ontology-linked
  //    entities to walk usefully for the smart arm.
  const grammar = (await client.buildGrammar(bonfireId)) as { entities?: number };
  if ((grammar.entities ?? 0) === 0) {
    console.warn(
      "bonfires provider: build_grammar returned 0 entities — smart arm will degrade to cascade-less search",
    );
  }
}
