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
    if (item.kind !== "delve_payload") return result
    const metadata = item.metadata
    const metadataRecord =
      metadata && typeof metadata === "object" && !Array.isArray(metadata)
        ? (metadata as Record<string, unknown>)
        : {}
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
