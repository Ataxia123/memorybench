import { createHash } from "node:crypto"
import type { UnifiedMessage, UnifiedSession } from "../../types/unified.js"
import type { BonfiresClient } from "./client.js"
import type { StackMessage } from "./types.js"
import { serializeSession } from "./ingest.js"

/** Minimal generic fallback types that graphiti's extractor recognizes
 * from pretraining. These catch concepts that don't fit the LLM-derived
 * specific types (Adoption_Placement_Record / Counseling_Program / etc.)
 * so entities aren't left bare :Entity. DateTime is deliberately omitted
 * — graphiti's extract_message / extract_text prompt filters temporal
 * info at extraction time ("do NOT extract dates, times, or years") and
 * surfaces it instead as `valid_at` edge properties via its temporal
 * resolver. */
const GENERIC_FALLBACK_TYPES: Array<{ name: string; description: string; parent_l1: string }> = [
  {
    name: "Person",
    description:
      "A specific named individual — speaker, acquaintance, family member, public figure. Only extract when named (or uniquely qualifiable, e.g. 'Nisha's dad'), never bare kinship terms alone.",
    parent_l1: "Person",
  },
  {
    name: "Event",
    description:
      "A discrete named occurrence — conference, ceremony, trip, party, performance, public event. Extract only when the event is uniquely identifiable (named or distinctly described).",
    parent_l1: "Event",
  },
  {
    name: "Place",
    description:
      "A named physical location — city, neighborhood, venue, park, building, country. Must be specific (not 'the city' or 'home').",
    parent_l1: "Location",
  },
  {
    name: "Organization",
    description:
      "A named collective — company, school, club, support group, nonprofit, movement, activist collective.",
    parent_l1: "Organization",
  },
  {
    name: "Activity",
    description:
      "A named recurring pursuit or practice — running, painting, pottery, cycling, counseling, mentorship. Specific enough that a reader would recognize the practice.",
    parent_l1: "Activity",
  },
  {
    name: "Object",
    description:
      "A specific distinguishable physical artifact — a painting with a title, an heirloom with provenance, a gift with a described property. Not generic nouns.",
    parent_l1: "Item",
  },
]

export function stackMessagesForSession(
  session: UnifiedSession,
  speakerSalt: string
): StackMessage[] {
  const referenceTime = session.metadata?.date as string | undefined
  const base = referenceTime ? Date.parse(referenceTime) : Date.now()
  const msgs = session.messages as UnifiedMessage[]
  return msgs.map((m, i) => {
    const metadata = {
      ...(m.metadata ?? {}),
      preserve_messages: true,
    }
    const id = stackMessageId(session.sessionId, i, metadata)
    metadata.message_id = id
    return {
      id,
      text: m.content,
      userId: speakerToUserId(m.speaker ?? m.role, speakerSalt),
      chatId: session.sessionId,
      sessionId: session.sessionId,
      timestamp: m.timestamp ?? new Date(base + i * 120_000).toISOString(),
      role: m.role,
      username: m.speaker,
      metadata,
    }
  })
}

function stackMessageId(
  sessionId: string,
  index: number,
  metadata: Record<string, unknown>
): string {
  const explicit = metadata.message_id
  if (typeof explicit === "string" && explicit.trim()) {
    return explicit
  }
  const sourceMessageId = metadata.source_message_id
  const sourceKind = metadata.source_kind
  if (
    sourceKind === "image_context" &&
    typeof sourceMessageId === "string" &&
    sourceMessageId.trim()
  ) {
    if (sourceMessageId.endsWith("-image")) {
      return sourceMessageId
    }
    return `${sourceMessageId}-image`
  }
  if (typeof sourceMessageId === "string" && sourceMessageId.trim()) {
    return sourceMessageId
  }
  return `${sessionId}-m${index}`
}

/**
 * Post-ingest indexing pipeline, ontology-guided-extraction via
 * `createEpisodeDirect` (no stack summarization layer):
 *
 *   1. startSummaries  → waitForJob      (per-chunk abstracts in Labeled_chunks)
 *   2. startTaxonomy   → waitForJob      (taxonomy labels from content)
 *   3. buildGrammar (discarded)          (side effect: derive_from_taxonomy
 *                                         → Mongo Ontology doc exists)
 *   4. per-session createEpisodeDirect with raw transcript as episode_body
 *      — the arq task loads the bonfire's Ontology, converts via
 *      `Ontology.to_graphiti_entity_types()`, and passes the dict to
 *      `graphiti.add_episode(entity_types=...)`. graphiti performs
 *      summary + entity + edge extraction in one structured LLM call
 *      over the RAW dialog. Extracted :Entity nodes carry ontology
 *      labels as Neo4j multi-labels (:Entity:Artwork / etc).
 *      Session date (`reference_time`) pins `valid_at` for temporal
 *      edges.
 *   5. buildCommunities sync             (Leiden over ontology-typed entities)
 *   6. buildOntology({ linkMethod: "structural", extendGrammar: "locomo" })
 *      — dominant-ontology-per-community via HAS_MEMBER label counts,
 *      plus grammar cascade rebuild.
 *
 * stack_process is intentionally skipped — its `_extract_episode_with_llm`
 * summarization step abstracts away verbatim claims (dates, quotes) that
 * LoCoMo questions depend on. graphiti's own extraction (with ontology
 * guidance) works over raw transcript and preserves the details.
 */
/** Stable per-speaker userId (SHA-1 prefix, optional salt). Lets stack_service's
 *  ``_build_user_context`` treat repeated ingestions of the same LoCoMo speaker
 *  across sessions as one actor — not that graphiti uses it today for
 *  cross-session entity dedup, but it keeps the data clean. */
function speakerToUserId(speaker: string, salt: string = ""): string {
  const hash = createHash("sha1").update(`${salt}:${speaker}`).digest("hex")
  return `lm-${hash.slice(0, 12)}`
}

export async function runIndexingPipeline(args: {
  client: BonfiresClient
  agentId: string
  bonfireId: string
  sessions: UnifiedSession[]
}): Promise<void> {
  const { client, agentId, bonfireId, sessions } = args

  // BONFIRES_PLAIN=1 skips taxonomy/ontology/communities/grammar entirely,
  // running the zep-baseline: per-message graphiti episodes with native
  // extraction (no entity_types guidance, no structural grammar cascade).
  // Still does summaries so the vector path has per-chunk context for the
  // 5-vector slot of the smart arm.
  const PLAIN = process.env.BONFIRES_PLAIN === "1"

  // BONFIRES_STACK_V2=1 hands the per-session "embedding jutsus" — chunks
  // grammar rebuild, hub seeding, cascade propagation, community update —
  // off to delve's Stack V2 path. memorybench's job shrinks to:
  //   1. summaries + taxonomy + buildGrammar (one-time per-bonfire setup
  //      that V2 still expects to find pre-derived)
  //   2. push every session's messages into ONE stack with sessionId tags
  //   3. ONE stackProcess call → delve does Phase A (per-session label +
  //      synth) → Phase B (bonfire grammar rebuild + hubs + cascade) →
  //      Phase C (FIFO Graphiti add_episode with update_communities=True)
  //   4. (still external for now) buildOntology final pass — moves
  //      inside delve once the lazy ontology→community link lands.
  // Server must have STACK_V2_ENABLED=1 for this to behave correctly.
  // Default-on; opt out via BONFIRES_STACK_V2=0. STACK_V2 + NO_DOC together
  // route ingestion through delve's _process_stack_background which runs
  // Phase A (per-session label+synth) → Phase B (rebuild+cascade) → Phase C
  // (graphiti drain) → Tier 4 seed all in one shot. The bench-side
  // pre-stack setup chain becomes a no-op (correct: it was operating on an
  // empty bonfire when NO_DOC skipped the per-session ingestContent path).
  const STACK_V2 = process.env.BONFIRES_STACK_V2 !== "0"

  // BONFIRES_STACK_V2_NO_DOC=1 short-circuits the entire pre-stack
  // summaries→taxonomy→buildGrammar→setOntology chain. The bonfire has
  // no chunks yet (ingestContent was skipped), so summaries would fail
  // anyway. delve's stackProcess Phase A.0a creates the synth chunks +
  // synthetic Summary rows, then Phase A.0a.5 runs the taxonomy
  // workflow + derive_ontology inline (when STACK_V2_DERIVE_INLINE=1
  // is set on the server). After stackProcess returns the bonfire has
  // a fully derived ontology AND classified+enriched chunks — nothing
  // for memorybench to do here.
  const NO_DOC = process.env.BONFIRES_STACK_V2_NO_DOC !== "0"
  if (NO_DOC && STACK_V2) {
    console.log(
      "STACK_V2_NO_DOC: skipping pre-stack summaries/taxonomy/buildGrammar/setOntology — delve handles them inline in Phase A"
    )
  } else {
    const summaries = await client.startSummaries(bonfireId)
    await client.waitForJob(summaries.job_id, { kind: "summaries", timeoutSec: 1800 })
  }

  if (!PLAIN && !(NO_DOC && STACK_V2)) {
    // v27i4 ordering — dependency chain with zero redundant work:
    //   summaries (chunks→summaries)
    //   taxonomy  (summaries→taxonomy labels)
    //   buildGrammar → side effect: derive_from_taxonomy populates the
    //     Ontology doc (LLM disambiguates taxonomy labels into atomic
    //     entity types with steering descriptions).
    //   resynthesize → wipes chunks+summaries, re-runs SessionSynthesizer
    //     with baseline + ontology labels merged into GLiNER. Per-bonfire
    //     domain spans (Pet, Book_Title, Identity_Label, ...) now land in
    //     chunk metadata and the query-shaped preference chunk text.
    //   re-summaries (new chunks need abstracts for the taxonomy labeler)
    //   labelChunks  (categories now attached to the right chunks)
    //   buildChunksGrammar (trimtab rebuild over new chunks)
    //   propagateCascadeEmbeddings (Phase B over new rules)
    //
    //   BONFIRES_ONTOLOGY_RESYNTH=0 opts out — falls back to the pre-v27i4
    //   ordering (labelChunks+build+cascade BEFORE buildGrammar, no
    //   ontology-driven GLiNER labels).
    const taxonomy = await client.startTaxonomy(bonfireId)
    await client.waitForJob(taxonomy.job_id, { kind: "taxonomy", timeoutSec: 1800 })

    if (process.env.BONFIRES_ONTOLOGY_RESYNTH !== "0") {
      // Derive ontology BEFORE touching chunk-side artifacts so the
      // synthesizer gets the label set on its first (post-resynth) run.
      await client.buildGrammar(bonfireId)

      if (STACK_V2) {
        // V2 server runs SessionSynthesizer per session inside Phase A
        // (synthesize_stack_sessions). Skipping the bench-side bonfire-wide
        // resynth call removes the long synchronous HTTP request and the
        // heartbeat workaround from the critical path. The ontology is
        // already derived above (buildGrammar), so V2's per-session synth
        // picks up the same label set the bench used to pass to resynth.
        console.log("ontology resynth: skipped (STACK_V2 — server handles per-session synth)")
      } else {
        const resynth = await client.resynthesizeChunks(bonfireId)
        console.log(
          `ontology resynth: labels_used=${resynth.ontology_labels_used} ` +
            `session_docs=${resynth.session_docs} ` +
            `chunks_deleted=${resynth.chunks_deleted} ` +
            `summaries_deleted=${resynth.summaries_deleted} ` +
            `chunks_created=${resynth.chunks_created} ` +
            `by_type=${JSON.stringify(resynth.by_type)} ` +
            `ontology_hits=${JSON.stringify(resynth.ontology_span_hits)}`
        )
        // No re-summaries call: resynth seeded synthetic Summary rows
        // (summary.content = chunk.content) so labelChunks + backfill
        // run without a second LLM summarization pass.
      }
    } else {
      // Legacy ordering retained for A/B.
    }

    // V2 hands labelChunks + buildChunksGrammar + cascade propagation off
    // to delve (runs inside stackProcess Phase B). V1 keeps doing them
    // here as a separate pre-episode pass.
    if (!STACK_V2) {
      const labelChunks = await client.startLabelChunks(bonfireId)
      await client.waitForJob(labelChunks.job_id, { kind: "label_chunks", timeoutSec: 1800 })

      const chunksBuild = await client.buildChunksGrammar(bonfireId)
      console.log(
        `chunks grammar built: inserted=${chunksBuild.rules_inserted ?? 0} ` +
          `updated=${chunksBuild.rules_updated ?? 0} ` +
          `errors=${chunksBuild.error_count ?? 0}`
      )

      if (process.env.BONFIRES_CASCADE_EMBEDDINGS !== "0") {
        const lambda = parseFloat(process.env.BONFIRES_CASCADE_LAMBDA ?? "0.2")
        const cascade = await client.propagateCascadeEmbeddings(bonfireId, { lambda })
        console.log(
          `cascade embeddings propagated: labels=${cascade.labels_seen ?? 0} ` +
            `updated=${cascade.chunks_updated ?? 0} ` +
            `skipped_no_vector=${cascade.chunks_skipped_no_vector ?? 0} ` +
            `errors=${cascade.error_count ?? 0}`
        )
      }
    }

    // BONFIRES_SKIP_BUILD_GRAMMAR=1 → true flat-KG ablation path.
    // Graphiti will fall back to its default entity extraction without
    // any custom ontology, producing generic :Entity:Activity etc. nodes
    // (same shape as v26i7). Use for clean isolated tests where L2
    // subtypes need to be off on the KG side.
    //
    // Otherwise — when BONFIRES_ONTOLOGY_RESYNTH=0 but we still want
    // Graphiti to see the derived ontology — run buildGrammar if it
    // wasn't already run in the main branch.
    if (
      process.env.BONFIRES_SKIP_BUILD_GRAMMAR !== "1" &&
      process.env.BONFIRES_ONTOLOGY_RESYNTH === "0"
    ) {
      await client.buildGrammar(bonfireId)
    }

    if (process.env.BONFIRES_SKIP_BUILD_GRAMMAR !== "1") {
      const current = await client.getOntology(bonfireId)
      const existingNames = new Set(current.entity_labels.map((l) => l.name))
      const merged = [
        ...current.entity_labels.map((l) => ({
          name: l.name,
          description: l.description,
          parent_l1: l.parent_l1 ?? null,
        })),
        ...GENERIC_FALLBACK_TYPES.filter((g) => !existingNames.has(g.name)),
      ]
      await client.setOntology(bonfireId, merged)
    }

    // Tier 4 spaCy seed: build dep_pattern_index from chunk.metadata.spacy_triples
    // (populated by spacy_enrich during resynthesizeChunks above) and MERGE
    // canonical :Entity + verb-typed edges into Neo4j BEFORE createEpisodeDirect
    // below. Order is load-bearing: graphiti's add_episode dedup uses fuzzy
    // name match against the entity space at extraction time, so seeding has
    // to land before per-episode LLM passes run. Default-on; opt out
    // via BONFIRES_TIER4_SEED=0 for clean A/B against pre-Tier4 runs.
    if (process.env.BONFIRES_TIER4_SEED !== "0") {
      const seed = await client.seedFromDepTree(bonfireId, { sync: true })
      console.log(
        `tier4 seed: triples_read=${seed.triples_read ?? 0} ` +
          `entities_seeded=${seed.entities_seeded ?? 0} ` +
          `edges_seeded=${seed.edges_seeded ?? 0} ` +
          `skipped_low_support=${seed.triples_skipped_low_support ?? 0}`
      )
    }
  }

  // Extraction granularity:
  //   - BONFIRES_EP_PER_SESSION=N (N=1..∞) splits each session into N
  //     roughly-equal chunks; each chunk becomes ONE graphiti episode
  //     with source="text" and the chunk's dialog as body. This matches
  //     the v12 "3 ep/session" path — bigger chunks, fewer episodes,
  //     cheaper ingest, richer per-episode extraction context.
  //   - Unset (default): per-message episodes (zep paper recipe). Each
  //     dialog turn is one episode with source="message" and the bare
  //     "speaker: content" format graphiti's message parser requires.
  const epPerSession = parseInt(process.env.BONFIRES_EP_PER_SESSION ?? "0", 10)
  // BONFIRES_INGEST_MODE=stack routes LoCoMo through the stack pipeline
  // (stack_service._extract_episode_with_llm runs + persists messages in
  // structured_content via the preserve_messages flag). Unset / "direct"
  // preserves the current v21 createEpisodeDirect path byte-for-byte.
  const ingestMode = (process.env.BONFIRES_INGEST_MODE ?? "direct") as "direct" | "stack"
  const speakerSalt = process.env.BONFIRES_SPEAKER_USER_ID_SALT ?? ""
  // BONFIRES_SKIP_EPISODES=1 short-circuits the per-session episode creation
  // loop. Used when the bonfire's graph was already built in a prior run and
  // we only want to re-run the chunks/grammar side (e.g., re-benching a new
  // synthesizer configuration against the same KG). Skips stackAdd +
  // stackProcess AND createEpisodeDirect; the rest of indexing (summaries,
  // taxonomy, label_chunks, buildGrammar, buildCommunities, buildOntology)
  // still runs.
  const skipEpisodes = process.env.BONFIRES_SKIP_EPISODES === "1"

  // Stack V2 wholesale path: push every session's messages into ONE
  // stack tagged with sessionId, then call stackProcess once. delve V2
  // partitions internally by sessionId, runs Phase A per session, then
  // does ONE bonfire-level grammar rebuild before draining episodes
  // FIFO into Graphiti. Bypasses every per-session loop below.
  if (STACK_V2 && !skipEpisodes) {
    const allStackMessages: StackMessage[] = []
    const stackPayloads: Array<{
      batch_messages: StackMessage[]
      user_updates: Array<Record<string, unknown>>
      batch_idx: number
    }> = []
    for (const session of sessions) {
      const batchMessages = stackMessagesForSession(session, speakerSalt)
      allStackMessages.push(...batchMessages)
      stackPayloads.push({
        batch_messages: batchMessages,
        user_updates: [],
        batch_idx: stackPayloads.length,
      })
    }
    if (process.env.BONFIRES_ARM === "hypermem" && process.env.BONFIRES_HYPERMEM_DIRECT_INDEX !== "0") {
      console.log(
        `hypermem direct stack index: draining ${allStackMessages.length} messages across ${stackPayloads.length} sessions`
      )
      await client.hypermemStackIndex({
        bonfireId,
        profile: process.env.BONFIRES_HYPERMEM_PROFILE ?? "nlp_taxonomy_v1",
        stackPayloads,
        initialCandidates: parseInt(process.env.BONFIRES_HYPERMEM_INITIAL_CANDIDATES ?? "100", 10),
        topicTopK: parseInt(process.env.BONFIRES_HYPERMEM_TOPIC_TOP_K ?? "15", 10),
        episodeTopK: parseInt(process.env.BONFIRES_HYPERMEM_EPISODE_TOP_K ?? "20", 10),
        factTopK: parseInt(process.env.BONFIRES_HYPERMEM_FACT_TOP_K ?? "30", 10),
        outputType: process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE ?? "011",
        useReranker: process.env.BONFIRES_HYPERMEM_RERANKER !== "0",
      })
      return
    }
    console.log(
      `stack V2: pushing ${allStackMessages.length} messages across ${sessions.length} sessions to one stack`
    )
    await client.stackAdd(agentId, allStackMessages)
    const { task_id } = await client.stackProcess(agentId)
    await client.waitForJob(task_id, {
      kind: "stack_process",
      // V2 process_stack does Phase A + Phase B + Phase C synchronously,
      // so the call is much longer than V1 (which only did label + extract
      // + queue-defer add_episode). Cap higher.
      timeoutSec: 7200,
    })
  }

  for (const session of sessions) {
    if (skipEpisodes) continue
    if (STACK_V2) continue // already handled wholesale above
    const referenceTime = session.metadata?.date as string | undefined
    const msgs = session.messages as Array<{ role: string; content: string; speaker?: string }>

    if (epPerSession > 0) {
      // Chunked mode: split msgs into N roughly-equal segments.
      const chunkSize = Math.ceil(msgs.length / epPerSession)
      for (let c = 0; c < epPerSession; c++) {
        const chunk = msgs.slice(c * chunkSize, (c + 1) * chunkSize)
        if (chunk.length === 0) continue

        if (ingestMode === "stack") {
          // Stack path: synthesize per-message timestamps (LoCoMo gives only
          // session-level dates), build StackMessage[], push via stackAdd
          // 2-at-a-time (backend limit), then stackProcess + waitForJob.
          // Each stackProcess call consumes all messages pushed since the
          // last process — one episode per chunk. preserve_messages=true on
          // every StackMessage tells delve stack_service to stamp
          // source="json" + persist raw messages in structured_content.
          const base = referenceTime ? Date.parse(referenceTime) : Date.now()
          const stackMessages: StackMessage[] = chunk.map((m, i) => {
            const globalIdx = c * chunkSize + i
            return {
              id: `${session.sessionId}-c${c}-m${i}`,
              text: m.content,
              userId: speakerToUserId(m.speaker ?? m.role, speakerSalt),
              chatId: session.sessionId,
              timestamp: new Date(base + globalIdx * 120_000).toISOString(),
              role: m.role,
              username: m.speaker,
              metadata: { preserve_messages: true },
            }
          })
          await client.stackAdd(agentId, stackMessages)
          const { task_id } = await client.stackProcess(agentId)
          await client.waitForJob(task_id, { kind: "stack_process", timeoutSec: 1800 })
        } else {
          const body = chunk.map((m) => `[${m.speaker ?? m.role}]: ${m.content}`).join("\n")
          const { task_id } = await client.createEpisodeDirect({
            bonfireId,
            name: `${session.sessionId}-c${c}`,
            episodeBody: body,
            referenceTime,
          })
          await client.waitForJob(task_id, {
            kind: "direct_episode",
            timeoutSec: 1800,
          })
        }
      }
    } else {
      // Per-message mode.
      for (let i = 0; i < msgs.length; i++) {
        const m = msgs[i]
        const body = `${m.speaker ?? m.role}: ${m.content}`
        const { task_id } = await client.createEpisodeDirect({
          bonfireId,
          name: `${session.sessionId}-m${i}`,
          episodeBody: body,
          referenceTime,
        })
        await client.waitForJob(task_id, {
          kind: "direct_episode",
          timeoutSec: 1800,
        })
      }
    }
  }

  if (!PLAIN) {
    // V2 sets update_communities=True on every Graphiti add_episode, so
    // community membership is maintained per-episode and the standalone
    // buildCommunities pass is a no-op (just refreshes Leiden topology
    // post-hoc, which the per-episode dirty-marker pattern handles).
    if (!STACK_V2) {
      await client.buildCommunities(bonfireId)
    }

    // buildOntology runs _link_ontology_to_taxonomies which uses Neo4j
    // vector.similarity.cosine() across taxonomy + ontology embeddings.
    // Pre-existing dimension-mismatch bug when embeddings from different
    // runs coexist. Wrapped in try/catch — the ontology linkage in Neo4j
    // is optional for retrieval (chunks_search doesn't use :OntologyLabel;
    // smart_unified's nodes/edges/chunks fan-out is unaffected). If the
    // underlying bug gets fixed we can drop the catch.
    //
    // STACK_V2 gate: when V2 is on, delve's pre-drain seeding has already
    // MERGEd (:Entity:{Type})-[:IS_A]->(:Entity:OntologyLabel:{Type})
    // triplets from the GLiNER engram aggregate. OntologyService.build()
    // wipes those hubs and rebuilds from cosine clustering, destroying
    // our seeded set (and the Tier 4 + graphiti dedup that hangs off it).
    // V2's post-graph chain handles ontology projection inline; the
    // bench-side build call is redundant and destructive.
    if (!STACK_V2) {
      try {
        await client.buildOntology(bonfireId, {
          linkMethod: "structural",
          extendGrammar: "locomo",
          grammarMinMentions: 1,
          grammarMinRelations: 1,
        })
      } catch (err) {
        console.warn(`buildOntology failed for ${bonfireId} (non-fatal): ${err}`)
      }
    }
  }
}
