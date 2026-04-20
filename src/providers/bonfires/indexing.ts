import type { BonfiresClient } from "./client.js";

/**
 * Run the full post-ingest indexing pipeline for a bonfire.
 *
 * Order:
 *   startSummaries  → waitForJob
 *   startTaxonomy   → waitForJob   (capture taxonomy run_id from job result)
 *   updateLabels    (POST /update_labels → creates KG entity + saves uuid
 *                   back to Taxonomy.uuid; Delve's primary mechanism for
 *                   populating taxonomy uuids)
 *   buildCommunities
 *   buildGrammar    (POST /trimtabs/grammars/{id}/build → runs
 *                   GrammarBuilderGraphCoordinator which reads the
 *                   now-uuid-populated taxonomies to build expansions)
 *
 * We skip `/labeling/hybrid` — it labels CHUNKS (for retrieval filtering),
 * not taxonomies. Its internal fallback to `bonfire.run_refs[-1]` for
 * update_labels_for_run uses the wrong run_id (a stack run_ref, not the
 * taxonomy run_ref), so uuid-setting silently fails. We call update_labels
 * directly with the correct taxonomy run_id instead.
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
  const taxonomyJob = await client.waitForJob(taxonomy.job_id, {
    kind: "taxonomy",
    timeoutSec: 1800,
  });

  // The API wraps the actual job output in metadata.result — not result directly.
  const taxonomyResult = (taxonomyJob.metadata as { result?: { run_id?: string } } | undefined)
    ?.result;
  const taxonomyRunId = taxonomyResult?.run_id;
  if (!taxonomyRunId) {
    throw new Error(
      `taxonomy job ${taxonomy.job_id} returned no run_id in metadata.result — cannot update labels`,
    );
  }
  await client.updateLabels(bonfireId, taxonomyRunId);

  await client.buildCommunities(bonfireId);

  await client.buildGrammar(bonfireId);
}
