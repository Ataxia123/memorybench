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

function summarizeMemoryKernelForCheckpoint(
  memoryKernel: Record<string, unknown>
): Record<string, unknown> {
  const diagnostics = recordValue(memoryKernel.diagnostics)
  return {
    candidate_id: memoryKernel.candidate_id,
    family: memoryKernel.family,
    source_ids: memoryKernel.source_ids,
    metadata: summarizeMemoryKernelCandidateMetadata(recordValue(memoryKernel.metadata)),
    diagnostics: {
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
      construct_hydration: diagnostics.construct_hydration,
      graph_hydration: diagnostics.graph_hydration,
      construction_learning: diagnostics.construction_learning,
    },
  }
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

function summarizeFcgSelection(selection: Record<string, unknown>): Record<string, unknown> {
  return {
    enabled: selection.enabled,
    top_k: selection.top_k,
    threshold: selection.threshold,
    selected_count: selection.selected_count,
    construct_definition_count: selection.construct_definition_count,
    selected_manifests: Array.isArray(selection.selected_manifests)
      ? selection.selected_manifests.map((manifest) => {
          const manifestRecord = recordValue(manifest)
          return {
            item_id: manifestRecord.item_id,
            topic_id: manifestRecord.topic_id,
            taxonomy_label: manifestRecord.taxonomy_label,
            similarity: manifestRecord.similarity,
          }
        })
      : selection.selected_manifests,
  }
}

function summarizeStatusAttempts(value: Record<string, unknown>): Record<string, unknown> {
  return {
    status: value.status,
    attempt_count: value.attempt_count,
  }
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
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
        const resultFile = join(resultsDir, `${question.questionId}.json`)
        const resultData = {
          questionId: question.questionId,
          question: question.question,
          questionType: question.questionType,
          groundTruth: question.groundTruth,
          containerTag,
          timestamp: new Date().toISOString(),
          durationMs,
          results,
        }

        writeFileSync(resultFile, JSON.stringify(resultData, null, 2))

        checkpointManager.updatePhase(checkpoint, question.questionId, "search", {
          status: "completed",
          resultFile,
          results: slimResultsForCheckpoint(results),
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
