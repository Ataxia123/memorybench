import type { BonfiresClient } from "./client.js";

export async function runIndexingPipeline(args: {
  client: BonfiresClient;
  agentId: string;
  bonfireId: string;
}): Promise<void> {
  const { client, agentId, bonfireId } = args;

  const { task_id } = await client.stackProcess(agentId);
  await client.waitForJob(task_id, { kind: "stack_processing", timeoutSec: 1800 });

  const taxonomy = await client.startTaxonomy(bonfireId);
  await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 });

  const labeling = await client.startLabeling(bonfireId);
  await client.waitForJob(labeling.job_id, { kind: "labeling", timeoutSec: 1800 });

  const grammar = (await client.buildGrammar(bonfireId)) as { entities?: number };
  if ((grammar.entities ?? 0) === 0) {
    console.warn(
      "bonfires provider: build_grammar returned 0 entities — smart arm will degrade to cascade-less search",
    );
  }

  await client.buildCommunities(bonfireId);
}
