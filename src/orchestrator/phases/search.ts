import { writeFileSync, mkdirSync, existsSync } from "fs"
import { join } from "path"
import type { Provider } from "../../types/provider"
import type { Benchmark } from "../../types/benchmark"
import type { RunCheckpoint } from "../../types/checkpoint"
import { CheckpointManager } from "../checkpoint"
import { logger } from "../../utils/logger"
import { ConcurrentExecutor } from "../concurrent"
import { resolveConcurrency } from "../../types/concurrency"

export function slimResultsForCheckpoint(results: unknown[]): unknown[] {
  return results.map((result) => {
    if (!result || typeof result !== "object" || Array.isArray(result)) return result
    const item = result as Record<string, unknown>
    const metadata = item.metadata
    const metadataRecord =
      metadata && typeof metadata === "object" && !Array.isArray(metadata)
        ? (metadata as Record<string, unknown>)
        : {}
    const memoryKernel = metadataRecord.memory_kernel
    const memoryKernelRecord =
      memoryKernel && typeof memoryKernel === "object" && !Array.isArray(memoryKernel)
        ? (memoryKernel as Record<string, unknown>)
        : {}
    if (Object.keys(memoryKernelRecord).length > 0) {
      return {
        text: item.text,
        score: item.score ?? null,
        kind: item.kind,
        metadata: {
          source: metadataRecord.source,
          memory_kernel: summarizeMemoryKernelForCheckpoint(memoryKernelRecord),
        },
      }
    }

    if (item.kind !== "delve_payload") return result
    const payload = metadataRecord.delve_payload
    const payloadRecord =
      payload && typeof payload === "object" && !Array.isArray(payload)
        ? (payload as Record<string, unknown>)
        : {}
    return {
      text: item.text,
      score: item.score ?? null,
      kind: item.kind,
      metadata: {
        hypermem_diagnostics: metadataRecord.hypermem_diagnostics,
        delve_payload_summary: summarizeDelvePayloadForCheckpoint(payloadRecord),
      },
    }
  })
}

export function extractSearchDiagnosticsForCheckpoint(results: unknown[]): Record<string, unknown> {
  const attachedDiagnostics = recordValue(
    (results as unknown as Record<string, unknown>).diagnostics
  )
  const attachedMemoryKernel = summarizeMemoryKernelDiagnostics(
    recordValue(attachedDiagnostics.memory_kernel)
  )
  if (Object.keys(attachedMemoryKernel).length > 0) {
    return { memory_kernel: attachedMemoryKernel }
  }
  for (const result of results) {
    if (!result || typeof result !== "object" || Array.isArray(result)) continue
    const item = result as Record<string, unknown>
    const metadata = recordValue(item.metadata)
    const memoryKernel = recordValue(metadata.memory_kernel)
    const diagnostics = summarizeMemoryKernelDiagnostics(recordValue(memoryKernel.diagnostics))
    if (Object.keys(diagnostics).length > 0) {
      return { memory_kernel: diagnostics }
    }
  }
  return {}
}

function summarizeMemoryKernelForCheckpoint(
  memoryKernel: Record<string, unknown>
): Record<string, unknown> {
  const answerCandidates = summarizeAnswerCandidates(memoryKernel.answer_candidates)
  const summary: Record<string, unknown> = {
    candidate_id: memoryKernel.candidate_id,
    family: memoryKernel.family,
    source_ids: memoryKernel.source_ids,
    metadata: summarizeMemoryKernelCandidateMetadata(recordValue(memoryKernel.metadata)),
  }
  if (answerCandidates.length > 0) summary.answer_candidates = answerCandidates
  return summary
}

function summarizeMemoryKernelDiagnostics(
  diagnostics: Record<string, unknown>
): Record<string, unknown> {
  if (Object.keys(diagnostics).length === 0) return {}
  return pruneUndefined({
    store_ms: diagnostics.store_ms,
    embed_ms: diagnostics.embed_ms,
    hydrate_ms: diagnostics.hydrate_ms,
    search_ms: diagnostics.search_ms,
    candidate_count: diagnostics.candidate_count,
    scored_count: diagnostics.scored_count,
    surface_query_count: diagnostics.surface_query_count,
    timings_ms: diagnostics.timings_ms,
    surface_counts: diagnostics.surface_counts,
    query_embedding: diagnostics.query_embedding,
    fcg_selection: summarizeFcgSelection(recordValue(diagnostics.fcg_selection)),
    fcg_comprehend: summarizeStatusAttempts(recordValue(diagnostics.fcg_comprehend)),
    fcg_activation: diagnostics.fcg_activation,
    fcg_grammar_cache: diagnostics.fcg_grammar_cache,
    retrieval_work_order: diagnostics.retrieval_work_order,
    comprehension_ir: diagnostics.comprehension_ir,
    ecs_search: diagnostics.ecs_search,
    search_methods: diagnostics.search_methods,
    surface_rows: diagnostics.surface_rows,
    topology_authority: diagnostics.topology_authority,
    score_breakdown_top_k: diagnostics.score_breakdown_top_k,
    construct_hydration: diagnostics.construct_hydration,
    graph_hydration: diagnostics.graph_hydration,
    construction_learning: diagnostics.construction_learning,
    aggregate_expansion: diagnostics.aggregate_expansion,
    answer_candidates: (() => {
      const answerCandidates = summarizeAnswerCandidates(diagnostics.answer_candidates)
      return answerCandidates.length > 0 ? answerCandidates : undefined
    })(),
  })
}

function summarizeMemoryKernelCandidateMetadata(
  metadata: Record<string, unknown>
): Record<string, unknown> {
  return {
    source_kind: metadata.source_kind,
    evidence_anchor_type: metadata.evidence_anchor_type,
    statement_id: metadata.statement_id,
    resolved_statement_ids: metadata.resolved_statement_ids,
    source_message_ids: metadata.source_message_ids,
    episode_ids: metadata.episode_ids,
    topic_id: metadata.topic_id,
    title: metadata.title,
    timestamp: metadata.timestamp,
    confidence: metadata.confidence,
    resolution_status: metadata.resolution_status,
    resolution_path: metadata.resolution_path,
  }
}

function summarizeAnswerCandidates(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return []
  return value
    .filter(
      (item): item is Record<string, unknown> =>
        Boolean(item) && typeof item === "object" && !Array.isArray(item)
    )
    .slice(0, 16)
    .map(summarizeAnswerCandidate)
    .filter((candidate) => Object.keys(candidate).length > 0)
}

function summarizeAnswerCandidate(candidate: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const key of [
    "family",
    "kind",
    "status",
    "confidence",
    "completeness",
    "text",
    "answer_kind",
    "source_candidate_id",
    "source_rank",
    "statement_id",
    "answer_role",
    "source_family",
    "source_message_id",
    "evidence_tier",
    "value",
    "answer",
    "normalized_value",
  ]) {
    const value = candidate[key]
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      out[key] = value
    }
  }

  for (const [key, maxEntries] of [
    ["typed_cues", 8],
    ["answer_evidence", 12],
    ["support", 10],
    ["conflicts", 8],
  ] as const) {
    const compact = compactRecord(candidate[key], maxEntries)
    if (Object.keys(compact).length > 0) out[key] = compact
  }

  for (const key of ["evidence_refs", "artifact_refs"] as const) {
    const compact = compactPrimitiveArray(candidate[key], 12)
    if (compact.length > 0) out[key] = compact
  }

  return out
}

function compactRecord(value: unknown, maxEntries: number): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  const out: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value as Record<string, unknown>).slice(0, maxEntries)) {
    if (typeof item === "string" || typeof item === "number" || typeof item === "boolean") {
      out[key] = item
    } else if (Array.isArray(item)) {
      const compact = compactPrimitiveArray(item, 12)
      if (compact.length > 0) out[key] = compact
    } else if (item && typeof item === "object") {
      const nested = compactRecord(item, 8)
      if (Object.keys(nested).length > 0) out[key] = nested
    }
  }
  return out
}

function compactPrimitiveArray(value: unknown, maxItems: number): unknown[] {
  if (!Array.isArray(value)) return []
  return value
    .filter(
      (item) => typeof item === "string" || typeof item === "number" || typeof item === "boolean"
    )
    .slice(0, maxItems)
}

function summarizeFcgSelection(selection: Record<string, unknown>): Record<string, unknown> {
  return pruneUndefined({
    enabled: selection.enabled,
    top_k: selection.top_k,
    threshold: selection.threshold,
    selected_count: selection.selected_count,
    construct_definition_count: selection.construct_definition_count,
    selection_mode: selection.selection_mode,
    route_registry_source: selection.route_registry_source,
    corpus_source: selection.corpus_source,
    selected_manifests: Array.isArray(selection.selected_manifests)
      ? selection.selected_manifests.map((manifest) => {
          const manifestRecord = recordValue(manifest)
          return pruneUndefined({
            item_id: manifestRecord.item_id,
            manifest_id: manifestRecord.manifest_id,
            topic_id: manifestRecord.topic_id,
            taxonomy_label: manifestRecord.taxonomy_label,
            score: manifestRecord.score ?? manifestRecord.similarity,
            similarity: manifestRecord.similarity,
            route_hits: manifestRecord.route_hits,
            route_keys: manifestRecord.route_keys,
            route_construct_count: manifestRecord.route_construct_count,
            active_route_construct_count: manifestRecord.active_route_construct_count,
            construct_definition_ids: compactStringArray(manifestRecord.construct_definition_ids),
            dag_status: manifestRecord.dag_status,
            episode_frontier_size: manifestRecord.episode_frontier_size,
            utterance_frontier_size: manifestRecord.utterance_frontier_size,
            occurrence_frontier_size: manifestRecord.occurrence_frontier_size,
          })
        })
      : selection.selected_manifests,
  })
}

function summarizeStatusAttempts(value: Record<string, unknown>): Record<string, unknown> {
  return pruneUndefined({
    status: value.status,
    attempt_count: value.attempt_count,
    query_activation_mode: value.query_activation_mode,
    query_supplier_mode: value.query_supplier_mode,
    fcg_activation_scope: (() => {
      const scope = compactRecord(value.fcg_activation_scope, 32)
      return Object.keys(scope).length > 0 ? scope : undefined
    })(),
    attempts: Array.isArray(value.attempts)
      ? value.attempts.slice(0, 8).map((attempt) => {
          const attemptRecord = recordValue(attempt)
          return pruneUndefined({
            status: attemptRecord.status,
            attempt_id: attemptRecord.attempt_id,
            item_id: attemptRecord.item_id,
            manifest_id: attemptRecord.manifest_id,
            construct_definition_count: attemptRecord.construct_definition_count,
            active_construct_definition_count: attemptRecord.active_construct_definition_count,
            candidate_count: attemptRecord.candidate_count,
            surface_query_count: attemptRecord.surface_query_count,
            elapsed_ms: attemptRecord.elapsed_ms,
            timings_ms: attemptRecord.timings_ms,
            fcg_activation_scope: (() => {
              const scope = compactRecord(attemptRecord.fcg_activation_scope, 32)
              return Object.keys(scope).length > 0 ? scope : undefined
            })(),
            error: attemptRecord.error,
          })
        })
      : value.attempts,
  })
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function compactStringArray(value: unknown, maxItems: number = 50): unknown {
  if (!Array.isArray(value)) return value
  return value.filter((item): item is string => typeof item === "string").slice(0, maxItems)
}

function pruneUndefined(record: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([, value]) => value !== undefined))
}

function summarizeDelvePayloadForCheckpoint(
  payload: Record<string, unknown>
): Record<string, unknown> {
  const envelope = payload.answer_context_envelope
  const envelopeRecord =
    envelope && typeof envelope === "object" && !Array.isArray(envelope)
      ? (envelope as Record<string, unknown>)
      : {}
  return {
    bonfire_id: payload.bonfire_id,
    profile: payload.profile,
    query: payload.query,
    counts: {
      topics: Array.isArray(payload.topics) ? payload.topics.length : 0,
      episodes: Array.isArray(payload.episodes) ? payload.episodes.length : 0,
      facts: Array.isArray(payload.facts) ? payload.facts.length : 0,
      evidence: Array.isArray(payload.evidence) ? payload.evidence.length : 0,
    },
    answer_context_envelope: {
      selected_answer_candidates: envelopeRecord.selected_answer_candidates,
      answer_candidates: envelopeRecord.answer_candidates,
      graph_hydration: envelopeRecord.graph_hydration,
    },
  }
}

export async function runSearchPhase(
  provider: Provider,
  benchmark: Benchmark,
  checkpoint: RunCheckpoint,
  checkpointManager: CheckpointManager,
  questionIds?: string[]
): Promise<void> {
  const questions = benchmark.getQuestions()
  const targetQuestions = questionIds
    ? questions.filter((q) => questionIds.includes(q.questionId))
    : questions

  const pendingQuestions = targetQuestions.filter((q) => {
    const status = checkpointManager.getPhaseStatus(checkpoint, q.questionId, "search")
    const indexingStatus = checkpointManager.getPhaseStatus(checkpoint, q.questionId, "indexing")
    return status !== "completed" && indexingStatus === "completed"
  })

  if (pendingQuestions.length === 0) {
    logger.info("No questions pending search")
    return
  }

  const resultsDir = checkpointManager.getResultsDir(checkpoint.runId)
  if (!existsSync(resultsDir)) {
    mkdirSync(resultsDir, { recursive: true })
  }

  const concurrency = resolveConcurrency("search", checkpoint.concurrency, provider.concurrency)

  logger.info(`Searching ${pendingQuestions.length} questions (concurrency: ${concurrency})...`)

  await ConcurrentExecutor.execute(
    pendingQuestions,
    concurrency,
    checkpoint.runId,
    "search",
    async ({ item: question, index, total }) => {
      const containerTag = `${question.questionId}-${checkpoint.dataSourceRunId}`

      const startTime = Date.now()
      checkpointManager.updatePhase(checkpoint, question.questionId, "search", {
        status: "in_progress",
        startedAt: new Date().toISOString(),
      })

      try {
        const results = await provider.search(question.question, {
          containerTag,
          limit: 10,
          threshold: 0.3,
        })

        const durationMs = Date.now() - startTime
        const diagnostics = extractSearchDiagnosticsForCheckpoint(results)
        const resultFile = join(resultsDir, `${question.questionId}.json`)
        const resultData = {
          questionId: question.questionId,
          question: question.question,
          questionType: question.questionType,
          groundTruth: question.groundTruth,
          containerTag,
          timestamp: new Date().toISOString(),
          durationMs,
          diagnostics,
          results,
        }

        writeFileSync(resultFile, JSON.stringify(resultData, null, 2))

        checkpointManager.updatePhase(checkpoint, question.questionId, "search", {
          status: "completed",
          resultFile,
          results: slimResultsForCheckpoint(results),
          diagnostics,
          completedAt: new Date().toISOString(),
          durationMs,
        })

        logger.progress(index + 1, total, `Searched ${question.questionId} (${durationMs}ms)`)
        return { questionId: question.questionId, durationMs }
      } catch (e) {
        const error = e instanceof Error ? e.message : String(e)
        checkpointManager.updatePhase(checkpoint, question.questionId, "search", {
          status: "failed",
          error,
        })
        logger.error(`Failed to search ${question.questionId}: ${error}`)
        throw new Error(
          `Search failed at ${question.questionId}: ${error}. Fix the issue and resume with the same run ID.`
        )
      }
    }
  )

  logger.success("Search phase complete")
}
