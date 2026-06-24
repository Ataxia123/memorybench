import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { dirname, join } from "path"
import { CheckpointManager } from "../../orchestrator/checkpoint"
import type { QuestionCheckpoint, RunCheckpoint } from "../../types/checkpoint"
import type { BenchmarkResult, EvaluationResult } from "../../types/unified"
import { CONV26_SMOKE_QUESTION_IDS } from "./ab-gate"

const RUNS_DIR = "./data/runs"

interface FailureMatrixArgs {
  runId: string
  baselineRunId?: string
  questionIds?: string[]
  excludeTypes: string[]
  smoke: boolean
  outputJson?: string
  outputMarkdown?: string
}

interface QuestionMatrixRow {
  questionId: string
  questionType: string
  score?: number
  baselineScore?: number
  hitAtK?: number
  answerHitRank?: number
  baselineHitAtK?: number
  baselineAnswerHitRank?: number
  resultCount: number
  searchDurationMs?: number
  topologyParticipated: boolean
  top1Family?: string
  top20FamilyCounts: Record<string, number>
  failureMode: string
  recovered: boolean
  regressed: boolean
}

interface FailureMatrixReport {
  runId: string
  baselineRunId?: string
  excludedTypes: string[]
  total: number
  correct: number
  accuracy: number
  baselineCorrect?: number
  recoveredCount: number
  regressionCount: number
  latencyMs: {
    mean?: number
    p95?: number
  }
  summary: Record<string, number>
  top1Families: Record<string, number>
  top20FamilyPresence: Record<string, number>
  rows: QuestionMatrixRow[]
}

export function parseArgs(args: string[]): FailureMatrixArgs | null {
  const parsed: Partial<FailureMatrixArgs> = { smoke: false, excludeTypes: [] }
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === "-r" || arg === "--run-id") {
      parsed.runId = args[++i]
    } else if (arg === "--baseline") {
      parsed.baselineRunId = args[++i]
    } else if (arg === "--questions") {
      parsed.questionIds = args[++i]
        ?.split(",")
        .map((item) => item.trim())
        .filter(Boolean)
    } else if (arg === "--exclude-type" || arg === "--exclude-types") {
      parsed.excludeTypes = [
        ...(parsed.excludeTypes ?? []),
        ...(args[++i]
          ?.split(",")
          .map((item) => item.trim())
          .filter(Boolean) ?? []),
      ]
    } else if (arg === "--smoke") {
      parsed.smoke = true
    } else if (arg === "--json") {
      parsed.outputJson = args[++i]
    } else if (arg === "--markdown" || arg === "--md") {
      parsed.outputMarkdown = args[++i]
    }
  }
  if (!parsed.runId) return null
  return {
    runId: parsed.runId,
    baselineRunId: parsed.baselineRunId,
    questionIds: parsed.questionIds,
    excludeTypes: parsed.excludeTypes ?? [],
    smoke: parsed.smoke ?? false,
    outputJson: parsed.outputJson,
    outputMarkdown: parsed.outputMarkdown,
  }
}

function readReport(runId: string): BenchmarkResult | undefined {
  const reportPath = join(RUNS_DIR, runId, "report.json")
  if (!existsSync(reportPath)) return undefined
  return JSON.parse(readFileSync(reportPath, "utf8")) as BenchmarkResult
}

function evaluationsByQuestion(runId: string): Map<string, EvaluationResult> {
  const report = readReport(runId)
  return new Map(
    (report?.evaluations ?? []).map((evaluation) => [evaluation.questionId, evaluation])
  )
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function answerHitRank(metrics: unknown): number | undefined {
  const record = recordValue(metrics)
  const explicit = numberValue(record.firstRelevantRank)
  if (explicit !== undefined && explicit > 0) return Math.round(explicit)

  const mrr = numberValue(record.mrr)
  if (mrr !== undefined && mrr > 0) {
    return Math.max(1, Math.round(1 / mrr))
  }

  const legacyHitAtK = numberValue(record.hitAtK)
  if (legacyHitAtK !== undefined && legacyHitAtK > 1) return Math.round(legacyHitAtK)
  if (legacyHitAtK === 1) return 1
  return undefined
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

function hasMeaningfulObject(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Object.keys(value as Record<string, unknown>).length > 0
}

function hasPositiveMetric(value: unknown, keys: string[]): boolean {
  const record = recordValue(value)
  return keys.some((key) => {
    const metric = record[key]
    return typeof metric === "number" && metric > 0
  })
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

function increment(record: Record<string, number>, key: string | undefined): void {
  if (!key) return
  record[key] = (record[key] ?? 0) + 1
}

function familyCounts(results: unknown[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const result of results.slice(0, 20)) {
    increment(out, resultFamily(result))
  }
  return out
}

export function failureMode(
  row: Omit<QuestionMatrixRow, "failureMode" | "recovered" | "regressed">
): string {
  if (row.score === 1) return "clear"
  if (row.resultCount === 0) return "retrieval_no_hit"
  if (row.answerHitRank === 1) return "answer_top1_hit"
  if (row.answerHitRank && row.answerHitRank > 1) return "answer_hit_not_top1"
  if (!row.hitAtK || row.hitAtK <= 0) return "retrieval_no_hit"
  if (row.hitAtK > 0) return "answer_hit_not_top1"
  return "clear"
}

function questionIdsFor(checkpoint: RunCheckpoint, args: FailureMatrixArgs): string[] {
  if (args.questionIds && args.questionIds.length > 0) return args.questionIds
  if (args.smoke) return CONV26_SMOKE_QUESTION_IDS
  return checkpoint.targetQuestionIds ?? Object.keys(checkpoint.questions)
}

function buildRow(
  question: QuestionCheckpoint,
  evaluation: EvaluationResult | undefined,
  baselineEvaluation: EvaluationResult | undefined
): QuestionMatrixRow {
  const search = question.phases.search
  const evaluate = question.phases.evaluate
  const results = Array.isArray(search.results)
    ? search.results
    : Array.isArray(evaluation?.searchResults)
      ? evaluation.searchResults
      : []
  const top20FamilyCounts = familyCounts(results)
  const retrievalMetrics = evaluate.retrievalMetrics ?? evaluation?.retrievalMetrics
  const baselineRetrievalMetrics = baselineEvaluation?.retrievalMetrics
  const base = {
    questionId: question.questionId,
    questionType: question.questionType,
    score: numberValue(evaluate.score) ?? numberValue(evaluation?.score),
    baselineScore: numberValue(baselineEvaluation?.score),
    hitAtK: numberValue(retrievalMetrics?.hitAtK),
    answerHitRank: answerHitRank(retrievalMetrics),
    baselineHitAtK: numberValue(baselineRetrievalMetrics?.hitAtK),
    baselineAnswerHitRank: answerHitRank(baselineRetrievalMetrics),
    resultCount: results.length,
    searchDurationMs: numberValue(search.durationMs) ?? numberValue(evaluation?.searchDurationMs),
    topologyParticipated: hasTopologyParticipation(search.diagnostics),
    top1Family: resultFamily(results[0]),
    top20FamilyCounts,
  }
  return {
    ...base,
    failureMode: failureMode(base),
    recovered: base.baselineScore !== undefined && base.baselineScore !== 1 && base.score === 1,
    regressed: base.baselineScore === 1 && base.score !== 1,
  }
}

function latencySummary(rows: QuestionMatrixRow[]): FailureMatrixReport["latencyMs"] {
  const values = rows
    .map((row) => row.searchDurationMs)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    .sort((a, b) => a - b)
  if (values.length === 0) return {}
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const index = Math.min(values.length - 1, Math.ceil(values.length * 0.95) - 1)
  return {
    mean: Math.round(mean),
    p95: values[index],
  }
}

export function buildFailureMatrixReport(args: FailureMatrixArgs): FailureMatrixReport {
  const checkpoint = new CheckpointManager().load(args.runId)
  if (!checkpoint) throw new Error(`No run found: ${args.runId}`)
  const evaluations = evaluationsByQuestion(args.runId)
  const baselineEvaluations = args.baselineRunId
    ? evaluationsByQuestion(args.baselineRunId)
    : new Map()
  const rows: QuestionMatrixRow[] = []
  const summary: Record<string, number> = {}
  const top1Families: Record<string, number> = {}
  const top20FamilyPresence: Record<string, number> = {}
  const excludedTypes = new Set(args.excludeTypes)

  for (const questionId of questionIdsFor(checkpoint, args)) {
    const question = checkpoint.questions[questionId]
    if (!question) continue
    if (excludedTypes.has(question.questionType)) continue
    const row = buildRow(question, evaluations.get(questionId), baselineEvaluations.get(questionId))
    rows.push(row)
    increment(summary, row.failureMode)
    increment(top1Families, row.top1Family)
    for (const family of Object.keys(row.top20FamilyCounts)) increment(top20FamilyPresence, family)
  }

  const correct = rows.filter((row) => row.score === 1).length
  const baselineCorrect = args.baselineRunId
    ? rows.filter((row) => row.baselineScore === 1).length
    : undefined
  return {
    runId: args.runId,
    baselineRunId: args.baselineRunId,
    excludedTypes: args.excludeTypes,
    total: rows.length,
    correct,
    accuracy: rows.length > 0 ? correct / rows.length : 0,
    baselineCorrect,
    recoveredCount: rows.filter((row) => row.recovered).length,
    regressionCount: rows.filter((row) => row.regressed).length,
    latencyMs: latencySummary(rows),
    summary,
    top1Families,
    top20FamilyPresence,
    rows,
  }
}

function renderMarkdown(report: FailureMatrixReport): string {
  const lines = [
    "# MemoryBench Failure Matrix",
    "",
    `Run: ${report.runId}`,
    report.baselineRunId ? `Baseline: ${report.baselineRunId}` : "",
    report.excludedTypes.length > 0 ? `Excluded types: ${report.excludedTypes.join(", ")}` : "",
    `Questions: ${report.total}`,
    `Accuracy: ${report.correct}/${report.total} (${(report.accuracy * 100).toFixed(2)}%)`,
    report.baselineRunId && report.baselineCorrect !== undefined
      ? `Baseline accuracy: ${report.baselineCorrect}/${report.total} (${((report.baselineCorrect / Math.max(1, report.total)) * 100).toFixed(2)}%)`
      : "",
    `Recovered: ${report.recoveredCount}`,
    `Regressed: ${report.regressionCount}`,
    `Search latency: mean=${report.latencyMs.mean ?? "?"}ms p95=${report.latencyMs.p95 ?? "?"}ms`,
    "",
    "## Summary",
    "",
    ...Object.entries(report.summary).map(([mode, count]) => `- ${mode}: ${count}`),
    "",
    "## Source Mix",
    "",
    `Top1: ${JSON.stringify(report.top1Families)}`,
    `Top20 presence: ${JSON.stringify(report.top20FamilyPresence)}`,
    "",
    "## Questions",
    "",
    "| Question | Type | Score | Hit@10 | Answer Rank | Baseline | Latency | Top1 | Mode | Delta | Top20 |",
    "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
  ].filter((line) => line !== "")

  for (const row of report.rows) {
    const baseline = [
      row.baselineScore === undefined ? "?" : String(row.baselineScore),
      row.baselineHitAtK === undefined ? "?" : String(row.baselineHitAtK),
      row.baselineAnswerHitRank === undefined ? "?" : `r${row.baselineAnswerHitRank}`,
    ].join("/")
    const delta = row.recovered ? "recovered" : row.regressed ? "regressed" : ""
    lines.push(
      `| ${row.questionId} | ${row.questionType} | ${row.score ?? "?"} | ${row.hitAtK ?? "?"} | ${row.answerHitRank ?? "?"} | ${baseline} | ${row.searchDurationMs ?? "?"} | ${row.top1Family ?? "none"} | ${row.failureMode} | ${delta} | ${JSON.stringify(row.top20FamilyCounts)} |`
    )
  }

  return `${lines.join("\n")}\n`
}

function writeArtifact(path: string, content: string): void {
  const directory = dirname(path)
  if (directory !== ".") mkdirSync(directory, { recursive: true })
  writeFileSync(path, content)
}

export async function failureMatrixCommand(args: string[]): Promise<void> {
  const parsed = parseArgs(args)
  if (!parsed) {
    console.log(
      "Usage: bun run src/index.ts failure-matrix -r <runId> [--baseline <runId>] [--exclude-type adversarial] [--smoke] [--questions q1,q2] [--json path] [--markdown path]"
    )
    return
  }

  const report = buildFailureMatrixReport(parsed)
  const markdown = renderMarkdown(report)
  console.log(markdown)

  if (parsed.outputJson) {
    writeArtifact(parsed.outputJson, JSON.stringify(report, null, 2))
  }
  if (parsed.outputMarkdown) {
    writeArtifact(parsed.outputMarkdown, markdown)
  }
}
