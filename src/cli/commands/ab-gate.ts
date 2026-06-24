import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { dirname, join } from "path"
import { CheckpointManager } from "../../orchestrator/checkpoint"
import type { RunCheckpoint, QuestionCheckpoint } from "../../types/checkpoint"
import type { BenchmarkResult, EvaluationResult, LatencyStats } from "../../types/unified"

const RUNS_DIR = "./data/runs"

export const CONV26_SMOKE_QUESTION_IDS = [
  "conv-26-q0",
  "conv-26-q15",
  "conv-26-q17",
  "conv-26-q20",
  "conv-26-q24",
  "conv-26-q42",
  "conv-26-q58",
  "conv-26-q61",
  "conv-26-q70",
  "conv-26-q81",
  "conv-26-q83",
  "conv-26-q92",
  "conv-26-q103",
  "conv-26-q104",
  "conv-26-q152",
  "conv-26-q158",
  "conv-26-q168",
  "conv-26-q178",
]

const SMOKE_ZERO_RESULT_SENTINELS = [
  "conv-26-q0",
  "conv-26-q15",
  "conv-26-q61",
  "conv-26-q70",
  "conv-26-q104",
]

const SMOKE_CONVERSION_SENTINELS = ["conv-26-q0", "conv-26-q58", "conv-26-q103", "conv-26-q152"]

const SMOKE_CORRECT_SENTINELS = [
  "conv-26-q17",
  "conv-26-q20",
  "conv-26-q24",
  "conv-26-q61",
  "conv-26-q70",
  "conv-26-q83",
  "conv-26-q92",
  "conv-26-q104",
  "conv-26-q168",
]

type GateMode = "smoke" | "full"

interface AbGateArgs {
  baselineRunId: string
  freshRunId: string
  mode: GateMode
  maxSearchP95Ms: number
  minTopologyCount: number
  outputJson?: string
  outputMarkdown?: string
}

interface QuestionSnapshot {
  questionId: string
  questionType?: string
  searchCompleted: boolean
  evaluateCompleted: boolean
  score?: number
  hitAtK?: number
  searchDurationMs?: number
  resultCount: number
  top1Family?: string
  top20Families: string[]
  topologyParticipated: boolean
}

interface RunSnapshot {
  runId: string
  source: "report+checkpoint" | "checkpoint"
  total: number
  searched: number
  evaluated: number
  correct: number
  hitAtK: number
  hitMeasured: number
  emptySearches: number
  failedSearches: number
  failedEvaluations: number
  topologyCount: number
  searchLatency?: LatencyStats
  top1Families: Record<string, number>
  top20FamilyPresence: Record<string, number>
  missingQuestionIds: string[]
  questions: Record<string, QuestionSnapshot>
}

interface AbGateReport {
  mode: GateMode
  baseline: RunSnapshot
  fresh: RunSnapshot
  comparison: {
    correctDelta: number
    hitAtKDelta: number
    searchP95DeltaMs?: number
    topologyDelta: number
  }
  failures: string[]
}

function parseNumber(value: string | undefined): number | undefined {
  if (value === undefined) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function parseArgs(args: string[]): AbGateArgs | null {
  const parsed: Partial<AbGateArgs> = {
    mode: "full",
    maxSearchP95Ms: 5000,
  }

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === "--baseline") {
      parsed.baselineRunId = args[++i]
    } else if (arg === "--fresh") {
      parsed.freshRunId = args[++i]
    } else if (arg === "--mode") {
      const mode = args[++i]
      if (mode !== "smoke" && mode !== "full") return null
      parsed.mode = mode
    } else if (arg === "--max-search-p95-ms") {
      const maxSearchP95Ms = parseNumber(args[++i])
      if (maxSearchP95Ms === undefined) return null
      parsed.maxSearchP95Ms = maxSearchP95Ms
    } else if (arg === "--min-topology-count") {
      const minTopologyCount = parseNumber(args[++i])
      if (minTopologyCount === undefined) return null
      parsed.minTopologyCount = minTopologyCount
    } else if (arg === "--json") {
      parsed.outputJson = args[++i]
    } else if (arg === "--markdown" || arg === "--md") {
      parsed.outputMarkdown = args[++i]
    }
  }

  if (!parsed.baselineRunId || !parsed.freshRunId) return null
  return {
    baselineRunId: parsed.baselineRunId,
    freshRunId: parsed.freshRunId,
    mode: parsed.mode ?? "full",
    maxSearchP95Ms: parsed.maxSearchP95Ms ?? 5000,
    minTopologyCount: parsed.minTopologyCount ?? (parsed.mode === "smoke" ? 2 : 1),
    outputJson: parsed.outputJson,
    outputMarkdown: parsed.outputMarkdown,
  }
}

function readReport(runId: string): BenchmarkResult | undefined {
  const reportPath = join(RUNS_DIR, runId, "report.json")
  if (!existsSync(reportPath)) return undefined
  return JSON.parse(readFileSync(reportPath, "utf8")) as BenchmarkResult
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

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function resultFamily(result: unknown): string | undefined {
  const resultRecord = recordValue(result)
  const metadata = recordValue(resultRecord.metadata)
  const memoryKernel = recordValue(metadata.memory_kernel)
  for (const value of [
    memoryKernel.family,
    recordValue(memoryKernel.metadata).evidence_anchor_type,
    resultRecord.kind,
    metadata.source,
  ]) {
    if (typeof value === "string" && value.length > 0) return value
  }
  return undefined
}

function hasPositiveMetric(value: unknown, keys: string[]): boolean {
  const record = recordValue(value)
  return keys.some((key) => {
    const metric = record[key]
    return typeof metric === "number" && metric > 0
  })
}

function hasMeaningfulObject(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Object.keys(value as Record<string, unknown>).length > 0
}

function hasTopologyParticipation(diagnostics: unknown): boolean {
  const memoryKernel = recordValue(recordValue(diagnostics).memory_kernel)
  const fcgSelection = recordValue(memoryKernel.fcg_selection)
  return (
    hasMeaningfulObject(memoryKernel.fcg_activation) ||
    hasMeaningfulObject(memoryKernel.retrieval_work_order) ||
    hasPositiveMetric(fcgSelection, ["selected_count", "construct_definition_count"]) ||
    hasPositiveMetric(memoryKernel.construct_hydration, [
      "hydrated_count",
      "occurrence_count",
      "construct_occurrence_count",
    ])
  )
}

function evaluationByQuestion(report: BenchmarkResult | undefined): Map<string, EvaluationResult> {
  return new Map(
    (report?.evaluations ?? []).map((evaluation) => [evaluation.questionId, evaluation])
  )
}

function questionSnapshot(
  question: QuestionCheckpoint,
  evaluation: EvaluationResult | undefined
): QuestionSnapshot {
  const search = question.phases.search
  const evaluate = question.phases.evaluate
  const results = Array.isArray(search.results)
    ? search.results
    : Array.isArray(evaluation?.searchResults)
      ? evaluation.searchResults
      : []
  const top20Families = results
    .slice(0, 20)
    .map(resultFamily)
    .filter((family): family is string => typeof family === "string")
  const score = numberValue(evaluate.score) ?? numberValue(evaluation?.score)
  const hitAtK =
    numberValue(evaluate.retrievalMetrics?.hitAtK) ??
    numberValue(evaluation?.retrievalMetrics?.hitAtK)
  const searchDurationMs =
    numberValue(search.durationMs) ?? numberValue(evaluation?.searchDurationMs)

  return {
    questionId: question.questionId,
    questionType: question.questionType,
    searchCompleted: search.status === "completed",
    evaluateCompleted: evaluate.status === "completed" || evaluation !== undefined,
    score,
    hitAtK,
    searchDurationMs,
    resultCount: results.length,
    top1Family: top20Families[0],
    top20Families,
    topologyParticipated: hasTopologyParticipation(search.diagnostics),
  }
}

function increment(record: Record<string, number>, key: string | undefined): void {
  if (!key) return
  record[key] = (record[key] ?? 0) + 1
}

function loadRunSnapshot(runId: string, questionIds?: string[]): RunSnapshot | null {
  const checkpoint = new CheckpointManager().load(runId)
  if (!checkpoint) return null
  const report = readReport(runId)
  const evaluations = evaluationByQuestion(report)
  const selectedQuestionIds =
    questionIds ?? checkpoint.targetQuestionIds ?? Object.keys(checkpoint.questions)
  const questions: Record<string, QuestionSnapshot> = {}
  const missingQuestionIds: string[] = []
  const top1Families: Record<string, number> = {}
  const top20FamilyPresence: Record<string, number> = {}

  for (const questionId of selectedQuestionIds) {
    const question = checkpoint.questions[questionId]
    if (!question) {
      missingQuestionIds.push(questionId)
      continue
    }
    const snapshot = questionSnapshot(question, evaluations.get(questionId))
    questions[questionId] = snapshot
    increment(top1Families, snapshot.top1Family)
    for (const family of new Set(snapshot.top20Families)) {
      increment(top20FamilyPresence, family)
    }
  }

  const questionValues = Object.values(questions)
  const searchDurations = questionValues
    .map((question) => question.searchDurationMs)
    .filter((duration): duration is number => duration !== undefined)

  return {
    runId,
    source: report ? "report+checkpoint" : "checkpoint",
    total: selectedQuestionIds.length,
    searched: questionValues.filter((question) => question.searchCompleted).length,
    evaluated: questionValues.filter((question) => question.evaluateCompleted).length,
    correct: questionValues.filter((question) => question.score === 1).length,
    hitAtK: questionValues.filter((question) => (question.hitAtK ?? 0) > 0).length,
    hitMeasured: questionValues.filter((question) => question.hitAtK !== undefined).length,
    emptySearches: questionValues.filter(
      (question) => question.searchCompleted && question.resultCount === 0
    ).length,
    failedSearches: Object.values(checkpoint.questions).filter(
      (question) => question.phases.search.status === "failed"
    ).length,
    failedEvaluations: Object.values(checkpoint.questions).filter(
      (question) => question.phases.evaluate.status === "failed"
    ).length,
    topologyCount: questionValues.filter((question) => question.topologyParticipated).length,
    searchLatency: calculateLatencyStats(searchDurations),
    top1Families,
    top20FamilyPresence,
    missingQuestionIds,
    questions,
  }
}

function evaluateSmokeGates(
  baseline: RunSnapshot,
  fresh: RunSnapshot,
  maxSearchP95Ms: number,
  minTopologyCount: number
): string[] {
  const failures: string[] = []
  if (fresh.missingQuestionIds.length > 0) {
    failures.push(`fresh run is missing smoke questions: ${fresh.missingQuestionIds.join(", ")}`)
  }
  if (fresh.evaluated !== CONV26_SMOKE_QUESTION_IDS.length) {
    failures.push(
      `fresh smoke evaluated ${fresh.evaluated}/${CONV26_SMOKE_QUESTION_IDS.length} questions`
    )
  }
  if (fresh.emptySearches > 0) {
    failures.push(`fresh smoke has ${fresh.emptySearches} empty search surfaces`)
  }
  for (const questionId of SMOKE_ZERO_RESULT_SENTINELS) {
    const resultCount = fresh.questions[questionId]?.resultCount ?? 0
    if (resultCount === 0) failures.push(`${questionId} has zero retrieved results`)
  }
  if (fresh.hitMeasured < CONV26_SMOKE_QUESTION_IDS.length) {
    failures.push(
      `fresh smoke only has Hit@K metrics for ${fresh.hitMeasured}/${CONV26_SMOKE_QUESTION_IDS.length} questions`
    )
  }
  if (fresh.hitAtK < 15) {
    failures.push(`fresh smoke Hit@10 ${fresh.hitAtK}/18 is below 15/18`)
  }
  const conversions = SMOKE_CONVERSION_SENTINELS.filter((questionId) => {
    const baselineHit = (baseline.questions[questionId]?.hitAtK ?? 0) > 0
    const freshHit = (fresh.questions[questionId]?.hitAtK ?? 0) > 0
    return !baselineHit && freshHit
  })
  if (conversions.length < 2) {
    failures.push(`fresh smoke converted ${conversions.length}/4 planned no-hit sentinels`)
  }
  if (fresh.correct < 10) {
    failures.push(`fresh smoke correct ${fresh.correct}/18 is below 10/18`)
  }
  for (const questionId of SMOKE_CORRECT_SENTINELS) {
    if (fresh.questions[questionId]?.score !== 1) {
      failures.push(`${questionId} did not remain correct`)
    }
  }
  if (!fresh.searchLatency) {
    failures.push("fresh smoke search latency is unavailable")
  } else if (fresh.searchLatency.p95 > maxSearchP95Ms) {
    failures.push(`fresh smoke p95 ${fresh.searchLatency.p95}ms exceeds ${maxSearchP95Ms}ms`)
  }
  if (fresh.topologyCount < minTopologyCount) {
    failures.push(
      `fresh smoke topology participation ${fresh.topologyCount} is below ${minTopologyCount}`
    )
  }
  if ((fresh.top1Families.index_doc ?? 0) < 6) {
    failures.push(
      `fresh smoke index_doc top1 count ${fresh.top1Families.index_doc ?? 0}/18 is below 6/18`
    )
  }
  if ((fresh.top1Families.statement_label ?? 0) > 12) {
    failures.push(
      `fresh smoke statement_label top1 count ${fresh.top1Families.statement_label ?? 0}/18 exceeds 12/18`
    )
  }
  if ((fresh.top20FamilyPresence.index_doc ?? 0) < 16) {
    failures.push(
      `fresh smoke index_doc top20 presence ${fresh.top20FamilyPresence.index_doc ?? 0}/18 is below 16/18`
    )
  }
  return failures
}

function evaluateFullGates(
  baseline: RunSnapshot,
  fresh: RunSnapshot,
  maxSearchP95Ms: number,
  minTopologyCount: number
): string[] {
  const failures: string[] = []
  if (fresh.total === 0) failures.push("fresh run has no selected questions")
  if (fresh.evaluated !== fresh.total) {
    failures.push(`fresh run evaluated ${fresh.evaluated}/${fresh.total} questions`)
  }
  if (fresh.emptySearches > 0) {
    failures.push(`fresh run has ${fresh.emptySearches} empty search surfaces`)
  }
  if (fresh.failedSearches > 0) {
    failures.push(`fresh run has ${fresh.failedSearches} failed searches`)
  }
  if (fresh.failedEvaluations > 0) {
    failures.push(`fresh run has ${fresh.failedEvaluations} failed evaluations`)
  }
  if (fresh.correct < baseline.correct) {
    failures.push(`fresh correct ${fresh.correct} regressed below baseline ${baseline.correct}`)
  }
  if (fresh.hitMeasured > 0 && baseline.hitMeasured > 0 && fresh.hitAtK < baseline.hitAtK) {
    failures.push(`fresh Hit@10 ${fresh.hitAtK} regressed below baseline ${baseline.hitAtK}`)
  }
  if (!fresh.searchLatency) {
    failures.push("fresh search latency is unavailable")
  } else if (fresh.searchLatency.p95 > maxSearchP95Ms) {
    failures.push(`fresh p95 ${fresh.searchLatency.p95}ms exceeds ${maxSearchP95Ms}ms`)
  }
  if (fresh.topologyCount < minTopologyCount) {
    failures.push(
      `fresh topology participation ${fresh.topologyCount} is below ${minTopologyCount}`
    )
  }
  return failures
}

export function buildAbGateReport(
  baseline: RunSnapshot,
  fresh: RunSnapshot,
  args: Pick<AbGateArgs, "mode" | "maxSearchP95Ms" | "minTopologyCount">
): AbGateReport {
  const failures =
    args.mode === "smoke"
      ? evaluateSmokeGates(baseline, fresh, args.maxSearchP95Ms, args.minTopologyCount)
      : evaluateFullGates(baseline, fresh, args.maxSearchP95Ms, args.minTopologyCount)

  return {
    mode: args.mode,
    baseline,
    fresh,
    comparison: {
      correctDelta: fresh.correct - baseline.correct,
      hitAtKDelta: fresh.hitAtK - baseline.hitAtK,
      searchP95DeltaMs:
        fresh.searchLatency && baseline.searchLatency
          ? fresh.searchLatency.p95 - baseline.searchLatency.p95
          : undefined,
      topologyDelta: fresh.topologyCount - baseline.topologyCount,
    },
    failures,
  }
}

function percent(correct: number, total: number): string {
  return total > 0 ? `${((correct / total) * 100).toFixed(2)}%` : "n/a"
}

function renderMarkdown(report: AbGateReport): string {
  const baseline = report.baseline
  const fresh = report.fresh
  const lines = [
    `# MemoryBench A/B Gate`,
    "",
    `Mode: ${report.mode}`,
    "",
    "| Run | Correct | Hit@10 | Empty | Topology | Search p95 |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    `| ${baseline.runId} | ${baseline.correct}/${baseline.total} (${percent(baseline.correct, baseline.total)}) | ${baseline.hitAtK}/${baseline.hitMeasured} | ${baseline.emptySearches} | ${baseline.topologyCount} | ${baseline.searchLatency?.p95 ?? "n/a"} |`,
    `| ${fresh.runId} | ${fresh.correct}/${fresh.total} (${percent(fresh.correct, fresh.total)}) | ${fresh.hitAtK}/${fresh.hitMeasured} | ${fresh.emptySearches} | ${fresh.topologyCount} | ${fresh.searchLatency?.p95 ?? "n/a"} |`,
    "",
    `Correct delta: ${report.comparison.correctDelta}`,
    `Hit@10 delta: ${report.comparison.hitAtKDelta}`,
    `Topology delta: ${report.comparison.topologyDelta}`,
    "",
    report.failures.length === 0 ? "Gate: PASS" : "Gate: FAIL",
  ]
  if (report.failures.length > 0) {
    lines.push("", "Failures:")
    for (const failure of report.failures) {
      lines.push(`- ${failure}`)
    }
  }
  return `${lines.join("\n")}\n`
}

function writeArtifact(path: string, content: string): void {
  const directory = dirname(path)
  if (directory !== ".") mkdirSync(directory, { recursive: true })
  writeFileSync(path, content)
}

function printUsage(): void {
  console.log(
    "Usage: bun run src/index.ts ab-gate --baseline <runId> --fresh <runId> [--mode smoke|full] [--json <path>] [--markdown <path>]"
  )
  console.log("")
  console.log("Options:")
  console.log("  --baseline             Baseline run ID")
  console.log("  --fresh                Fresh topology run ID")
  console.log("  --mode                 Gate mode: smoke or full (default: full)")
  console.log("  --max-search-p95-ms    Maximum fresh search p95 latency (default: 5000)")
  console.log("  --min-topology-count   Minimum questions with topology diagnostics")
  console.log("  --json                 Write JSON gate report")
  console.log("  --markdown, --md       Write Markdown gate report")
}

export async function abGateCommand(args: string[]): Promise<void> {
  const parsed = parseArgs(args)
  if (!parsed) {
    printUsage()
    return
  }

  const questionIds = parsed.mode === "smoke" ? CONV26_SMOKE_QUESTION_IDS : undefined
  const baseline = loadRunSnapshot(parsed.baselineRunId, questionIds)
  const fresh = loadRunSnapshot(parsed.freshRunId, questionIds)
  if (!baseline) {
    console.error(`No checkpoint found for baseline run: ${parsed.baselineRunId}`)
    process.exitCode = 1
    return
  }
  if (!fresh) {
    console.error(`No checkpoint found for fresh run: ${parsed.freshRunId}`)
    process.exitCode = 1
    return
  }

  const report = buildAbGateReport(baseline, fresh, parsed)
  const markdown = renderMarkdown(report)

  console.log(markdown.trimEnd())
  console.log("")
  console.log("Fresh top1 families:")
  for (const [family, count] of Object.entries(fresh.top1Families).sort(([a], [b]) =>
    a.localeCompare(b)
  )) {
    console.log(`  ${family}: ${count}`)
  }

  if (parsed.outputJson) {
    writeArtifact(parsed.outputJson, `${JSON.stringify(report, null, 2)}\n`)
  }
  if (parsed.outputMarkdown) {
    writeArtifact(parsed.outputMarkdown, markdown)
  }

  if (report.failures.length > 0) {
    for (const failure of report.failures) {
      console.error(`Gate failed: ${failure}`)
    }
    process.exitCode = 1
  }
}
