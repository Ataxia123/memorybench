import { existsSync, readFileSync } from "fs"
import { join } from "path"
import { CheckpointManager } from "../../orchestrator/checkpoint"
import type { RunCheckpoint } from "../../types/checkpoint"
import type {
  BenchmarkResult,
  EvaluationResult,
  LatencyStats,
  QuestionSliceStats,
} from "../../types/unified"

const RUNS_DIR = "./data/runs"

interface ReportGateArgs {
  runId: string
  metric: "ex-adversarial"
  min?: number
  minTypeAccuracy?: number
  maxSearchMeanMs?: number
  maxSearchMedianMs?: number
  maxSearchP95Ms?: number
}

interface GateReport {
  slice: QuestionSliceStats
  byQuestionType: Record<string, QuestionSliceStats>
  searchLatency?: LatencyStats
  source: "report" | "checkpoint"
}

function parseNumber(value: string | undefined): number | undefined {
  if (value === undefined) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function parseArgs(args: string[]): ReportGateArgs | null {
  const parsed: Partial<ReportGateArgs> = { metric: "ex-adversarial" }

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === "-r" || arg === "--run-id") {
      parsed.runId = args[++i]
    } else if (arg === "--metric") {
      const metric = args[++i]
      if (metric === "ex-adversarial") parsed.metric = metric
    } else if (arg === "--min") {
      parsed.min = parseNumber(args[++i])
    } else if (arg === "--min-type-accuracy") {
      parsed.minTypeAccuracy = parseNumber(args[++i])
    } else if (arg === "--max-search-mean-ms") {
      parsed.maxSearchMeanMs = parseNumber(args[++i])
    } else if (arg === "--max-search-median-ms") {
      parsed.maxSearchMedianMs = parseNumber(args[++i])
    } else if (arg === "--max-search-p95-ms") {
      parsed.maxSearchP95Ms = parseNumber(args[++i])
    }
  }

  if (!parsed.runId) return null
  return parsed as ReportGateArgs
}

function computeSliceFromEvaluations(evaluations: EvaluationResult[]): QuestionSliceStats {
  const slice = evaluations.filter((evaluation) => evaluation.questionType !== "adversarial")
  const correct = slice.filter((evaluation) => evaluation.score === 1).length
  return {
    total: slice.length,
    correct,
    accuracy: slice.length > 0 ? correct / slice.length : 0,
  }
}

function computeSliceFromCheckpoint(checkpoint: RunCheckpoint): QuestionSliceStats {
  const evaluated = Object.values(checkpoint.questions).filter(
    (question) =>
      question.questionType !== "adversarial" && question.phases.evaluate.status === "completed"
  )
  const correct = evaluated.filter((question) => question.phases.evaluate.score === 1).length
  return {
    total: evaluated.length,
    correct,
    accuracy: evaluated.length > 0 ? correct / evaluated.length : 0,
  }
}

export function computeTypeSlicesFromEvaluations(
  evaluations: EvaluationResult[],
  excludeTypes: string[] = ["adversarial"]
): Record<string, QuestionSliceStats> {
  const excluded = new Set(excludeTypes)
  const byType: Record<string, QuestionSliceStats> = {}
  for (const evaluation of evaluations) {
    if (excluded.has(evaluation.questionType)) continue
    const stats = (byType[evaluation.questionType] ??= { total: 0, correct: 0, accuracy: 0 })
    stats.total += 1
    if (evaluation.score === 1) stats.correct += 1
  }
  for (const stats of Object.values(byType)) {
    stats.accuracy = stats.total > 0 ? stats.correct / stats.total : 0
  }
  return byType
}

function computeTypeSlicesFromCheckpoint(checkpoint: RunCheckpoint): Record<string, QuestionSliceStats> {
  const evaluations: EvaluationResult[] = Object.values(checkpoint.questions)
    .filter((question) => question.phases.evaluate.status === "completed")
    .map((question) => ({
      questionId: question.questionId,
      questionType: question.questionType,
      question: question.question,
      score: question.phases.evaluate.score === 1 ? 1 : 0,
      label: question.phases.evaluate.score === 1 ? "correct" : "incorrect",
      explanation: question.phases.evaluate.explanation ?? "",
      hypothesis: question.phases.answer.hypothesis ?? "",
      groundTruth: question.groundTruth,
      searchResults: question.phases.search.results ?? [],
      searchDurationMs: question.phases.search.durationMs ?? 0,
      answerDurationMs: question.phases.answer.durationMs ?? 0,
      totalDurationMs: 0,
    }))
  return computeTypeSlicesFromEvaluations(evaluations)
}

function calculateLatencyStats(durations: number[]): LatencyStats | undefined {
  if (durations.length === 0) return undefined
  const sorted = [...durations].sort((a, b) => a - b)
  const n = sorted.length
  const sum = sorted.reduce((a, b) => a + b, 0)
  const mean = sum / n
  const variance = sorted.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n

  return {
    min: sorted[0],
    max: sorted[n - 1],
    mean: Math.round(mean),
    median: sorted[Math.floor(n / 2)],
    p95: sorted[Math.floor(n * 0.95)] || sorted[n - 1],
    p99: sorted[Math.floor(n * 0.99)] || sorted[n - 1],
    stdDev: Math.round(Math.sqrt(variance)),
    count: n,
  }
}

function loadGateReport(runId: string): GateReport | null {
  const reportPath = join(RUNS_DIR, runId, "report.json")
  if (existsSync(reportPath)) {
    const report = JSON.parse(readFileSync(reportPath, "utf8")) as BenchmarkResult
    return {
      slice: report.slices?.exAdversarial ?? computeSliceFromEvaluations(report.evaluations),
      byQuestionType: computeTypeSlicesFromEvaluations(report.evaluations),
      searchLatency: report.latency.search,
      source: "report",
    }
  }

  const checkpoint = new CheckpointManager().load(runId)
  if (!checkpoint) return null
  const searchDurations = Object.values(checkpoint.questions)
    .filter((question) => question.phases.search.status === "completed")
    .map((question) => question.phases.search.durationMs)
    .filter((duration): duration is number => typeof duration === "number")

  return {
    slice: computeSliceFromCheckpoint(checkpoint),
    byQuestionType: computeTypeSlicesFromCheckpoint(checkpoint),
    searchLatency: calculateLatencyStats(searchDurations),
    source: "checkpoint",
  }
}

function printUsage(): void {
  console.log("Usage: bun run src/index.ts report-gate -r <runId> [--min <score>] [latency gates]")
  console.log("")
  console.log("Options:")
  console.log("  -r, --run-id              Run identifier")
  console.log("  --metric                  Metric to gate, currently ex-adversarial")
  console.log("  --min                     Minimum ex-adversarial accuracy, e.g. 0.80")
  console.log("  --min-type-accuracy       Minimum per-type ex-adversarial accuracy, e.g. 0.60")
  console.log("  --max-search-mean-ms      Maximum mean search latency")
  console.log("  --max-search-median-ms    Maximum median search latency")
  console.log("  --max-search-p95-ms       Maximum p95 search latency")
}

export async function reportGateCommand(args: string[]): Promise<void> {
  const parsed = parseArgs(args)
  if (!parsed) {
    printUsage()
    return
  }

  const gateReport = loadGateReport(parsed.runId)
  if (!gateReport) {
    console.error(`No report or checkpoint found for run: ${parsed.runId}`)
    process.exitCode = 1
    return
  }

  const failures: string[] = []
  const slice = gateReport.slice
  if (parsed.min !== undefined && slice.accuracy < parsed.min) {
    failures.push(
      `ex-adversarial accuracy ${(slice.accuracy * 100).toFixed(2)}% is below ${(parsed.min * 100).toFixed(2)}%`
    )
  }
  if (parsed.minTypeAccuracy !== undefined) {
    for (const [questionType, stats] of Object.entries(gateReport.byQuestionType).sort(([a], [b]) =>
      a.localeCompare(b)
    )) {
      if (stats.accuracy < parsed.minTypeAccuracy) {
        failures.push(
          `${questionType} accuracy ${(stats.accuracy * 100).toFixed(2)}% is below ${(parsed.minTypeAccuracy * 100).toFixed(2)}%`
        )
      }
    }
  }

  const latency = gateReport.searchLatency
  if (latency) {
    if (parsed.maxSearchMeanMs !== undefined && latency.mean > parsed.maxSearchMeanMs) {
      failures.push(`search mean ${latency.mean}ms exceeds ${parsed.maxSearchMeanMs}ms`)
    }
    if (parsed.maxSearchMedianMs !== undefined && latency.median > parsed.maxSearchMedianMs) {
      failures.push(`search median ${latency.median}ms exceeds ${parsed.maxSearchMedianMs}ms`)
    }
    if (parsed.maxSearchP95Ms !== undefined && latency.p95 > parsed.maxSearchP95Ms) {
      failures.push(`search p95 ${latency.p95}ms exceeds ${parsed.maxSearchP95Ms}ms`)
    }
  }

  console.log(`Run: ${parsed.runId} (${gateReport.source})`)
  console.log(
    `Ex-adversarial: ${slice.correct}/${slice.total} (${(slice.accuracy * 100).toFixed(2)}%)`
  )
  if (latency) {
    console.log(
      `Search latency: mean=${latency.mean}ms median=${latency.median}ms p95=${latency.p95}ms count=${latency.count}`
    )
  }
  const typeEntries = Object.entries(gateReport.byQuestionType).sort(([a], [b]) => a.localeCompare(b))
  if (typeEntries.length > 0) {
    console.log("Ex-adversarial by type:")
    for (const [questionType, stats] of typeEntries) {
      console.log(`  ${questionType}: ${stats.correct}/${stats.total} (${(stats.accuracy * 100).toFixed(2)}%)`)
    }
  }

  if (failures.length > 0) {
    for (const failure of failures) {
      console.error(`Gate failed: ${failure}`)
    }
    process.exitCode = 1
  }
}
