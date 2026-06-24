import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { dirname, join } from "path"
import { CheckpointManager } from "../../orchestrator/checkpoint"
import type { QuestionCheckpoint, RunCheckpoint } from "../../types/checkpoint"
import type { BenchmarkResult, EvaluationResult, LatencyStats } from "../../types/unified"

const RUNS_DIR = "./data/runs"

export type NativeActivationGateStage = "smoke" | "probe" | "full"

export interface NativeActivationGateArgs {
  runId: string
  stage: NativeActivationGateStage
  questionIds?: string[]
  maxSearchP95Ms: number
  expectedTotal?: number
  minCorrect?: number
  smokeRunId?: string
  probeRunId?: string
  allowMissingPrereq: boolean
  outputJson?: string
  outputMarkdown?: string
}

export interface NativeActivationQuestionSnapshot {
  questionId: string
  questionType: string
  searchStatus: string
  evaluateStatus: string
  score?: number
  resultCount: number
  searchDurationMs?: number
  scopedActivationDiagnostics: boolean
  graphOnlyEvidence: boolean
  graphExpansionWithoutFrontier: boolean
  outOfFrontierGraphRows: number
  topFamilies: string[]
}

export interface NativeActivationRunSnapshot {
  runId: string
  stage: NativeActivationGateStage
  source: "report+checkpoint" | "checkpoint"
  total: number
  searched: number
  evaluated: number
  correct: number
  failedSearches: number
  emptySearches: number
  graphOnlyEvidence: number
  scopedDiagnosticsMissing: number
  graphExpansionWithoutFrontier: number
  outOfFrontierGraphRows: number
  searchLatency?: LatencyStats
  missingQuestionIds: string[]
  questions: Record<string, NativeActivationQuestionSnapshot>
}

export interface NativeActivationGateReport {
  stage: NativeActivationGateStage
  run: NativeActivationRunSnapshot
  prerequisites: NativeActivationRunSnapshot[]
  failures: string[]
}

function parseNumber(value: string | undefined): number | undefined {
  if (value === undefined) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function parseQuestionIds(value: string | undefined): string[] | undefined {
  if (!value) return undefined
  const ids = value
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
  return ids.length > 0 ? ids : undefined
}

export function parseNativeActivationGateArgs(args: string[]): NativeActivationGateArgs | null {
  const parsed: Partial<NativeActivationGateArgs> = {
    stage: "full",
    maxSearchP95Ms: 6000,
    allowMissingPrereq: false,
  }

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === "-r" || arg === "--run-id") {
      parsed.runId = args[++i]
    } else if (arg === "--stage" || arg === "--mode") {
      const stage = args[++i]
      if (stage !== "smoke" && stage !== "probe" && stage !== "full") return null
      parsed.stage = stage
    } else if (arg === "--questions") {
      parsed.questionIds = parseQuestionIds(args[++i])
    } else if (arg === "--max-search-p95-ms") {
      const value = parseNumber(args[++i])
      if (value === undefined) return null
      parsed.maxSearchP95Ms = value
    } else if (arg === "--expected-total") {
      const value = parseNumber(args[++i])
      if (value === undefined) return null
      parsed.expectedTotal = value
    } else if (arg === "--min-correct") {
      const value = parseNumber(args[++i])
      if (value === undefined) return null
      parsed.minCorrect = value
    } else if (arg === "--smoke-run") {
      parsed.smokeRunId = args[++i]
    } else if (arg === "--probe-run") {
      parsed.probeRunId = args[++i]
    } else if (arg === "--allow-missing-prereq") {
      parsed.allowMissingPrereq = true
    } else if (arg === "--json") {
      parsed.outputJson = args[++i]
    } else if (arg === "--markdown" || arg === "--md") {
      parsed.outputMarkdown = args[++i]
    }
  }

  if (!parsed.runId) return null
  const stage = parsed.stage ?? "full"
  return {
    runId: parsed.runId,
    stage,
    questionIds: parsed.questionIds,
    maxSearchP95Ms: parsed.maxSearchP95Ms ?? 6000,
    expectedTotal: parsed.expectedTotal ?? (stage === "full" ? 152 : undefined),
    minCorrect: parsed.minCorrect ?? (stage === "full" ? 106 : undefined),
    smokeRunId: parsed.smokeRunId,
    probeRunId: parsed.probeRunId,
    allowMissingPrereq: parsed.allowMissingPrereq ?? false,
    outputJson: parsed.outputJson,
    outputMarkdown: parsed.outputMarkdown,
  }
}

function readReport(runId: string): BenchmarkResult | undefined {
  const reportPath = join(RUNS_DIR, runId, "report.json")
  if (!existsSync(reportPath)) return undefined
  return JSON.parse(readFileSync(reportPath, "utf8")) as BenchmarkResult
}

function evaluationByQuestion(report: BenchmarkResult | undefined): Map<string, EvaluationResult> {
  return new Map(
    (report?.evaluations ?? []).map((evaluation) => [evaluation.questionId, evaluation])
  )
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

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value : undefined
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === "string" && item.length > 0)
}

function nonEmptyRecord(value: unknown): boolean {
  return Object.keys(recordValue(value)).length > 0
}

function resultFamily(result: unknown): string | undefined {
  const resultRecord = recordValue(result)
  const metadata = recordValue(resultRecord.metadata)
  const memoryKernel = recordValue(metadata.memory_kernel)
  const kernelMetadata = recordValue(memoryKernel.metadata)
  for (const value of [
    memoryKernel.family,
    kernelMetadata.evidence_anchor_type,
    kernelMetadata.source_kind,
    resultRecord.kind,
    metadata.source,
  ]) {
    const text = stringValue(value)
    if (text) return text
  }
  return undefined
}

function isGraphOnlyResult(result: unknown): boolean {
  const family = resultFamily(result)?.toLowerCase() ?? ""
  const resultRecord = recordValue(result)
  const metadata = recordValue(resultRecord.metadata)
  const memoryKernel = recordValue(metadata.memory_kernel)
  const kernelMetadata = recordValue(memoryKernel.metadata)
  const evidenceTier = stringValue(kernelMetadata.evidence_tier)?.toLowerCase() ?? ""
  const sourceKind = stringValue(kernelMetadata.source_kind)?.toLowerCase() ?? ""
  const source = stringValue(metadata.source)?.toLowerCase() ?? ""
  return (
    family.includes("graph") ||
    sourceKind.includes("graph") ||
    source.includes("graph") ||
    evidenceTier === "episode_context"
  )
}

function hasAnswerGradeCandidate(memoryKernel: Record<string, unknown>): boolean {
  const diagnosticsAnswerCandidates = recordValue(memoryKernel.diagnostics).answer_candidates
  const candidates: unknown[] = Array.isArray(memoryKernel.answer_candidates)
    ? memoryKernel.answer_candidates
    : Array.isArray(diagnosticsAnswerCandidates)
      ? diagnosticsAnswerCandidates
      : []
  return candidates.some((candidate) => {
    const record = recordValue(candidate)
    const evidenceTier = stringValue(record.evidence_tier)?.toLowerCase() ?? ""
    const status = stringValue(record.status)?.toLowerCase() ?? ""
    return evidenceTier !== "episode_context" && status !== "filtered"
  })
}

function memoryKernelDiagnostics(searchDiagnostics: unknown): Record<string, unknown> {
  return recordValue(recordValue(searchDiagnostics).memory_kernel)
}

function hasScopedActivationDiagnostics(memoryKernel: Record<string, unknown>): boolean {
  const fcgComprehend = recordValue(memoryKernel.fcg_comprehend)
  if (nonEmptyRecord(fcgComprehend.fcg_activation_scope)) return true
  const attempts = Array.isArray(fcgComprehend.attempts) ? fcgComprehend.attempts : []
  return attempts.some((attempt) => nonEmptyRecord(recordValue(attempt).fcg_activation_scope))
}

function graphExpansionWithoutFrontier(memoryKernel: Record<string, unknown>): boolean {
  const graph = recordValue(memoryKernel.graph_hydration)
  const hydratedCount =
    numberValue(graph.hydrated_count) ??
    numberValue(graph.graph_row_count) ??
    numberValue(graph.local_hydration_count) ??
    0
  const outOfFrontierRows = numberValue(graph.out_of_frontier_graph_rows) ?? 0
  if (hydratedCount <= 0 && outOfFrontierRows <= 0) return false
  const workOrder = recordValue(memoryKernel.retrieval_work_order)
  const frontier = recordValue(workOrder.episode_frontier)
  return stringArray(frontier.episode_ids).length === 0
}

function outOfFrontierGraphRows(memoryKernel: Record<string, unknown>): number {
  return numberValue(recordValue(memoryKernel.graph_hydration).out_of_frontier_graph_rows) ?? 0
}

function checkpointResults(
  question: QuestionCheckpoint,
  evaluation: EvaluationResult | undefined
): unknown[] {
  const search = question.phases.search
  if (Array.isArray(search.results)) return search.results
  if (Array.isArray(evaluation?.searchResults)) return evaluation.searchResults
  return []
}

function questionSnapshot(
  question: QuestionCheckpoint,
  evaluation: EvaluationResult | undefined
): NativeActivationQuestionSnapshot {
  const search = question.phases.search
  const evaluate = question.phases.evaluate
  const results = checkpointResults(question, evaluation)
  const diagnostics = memoryKernelDiagnostics(search.diagnostics)
  const resultCount = numberValue(search.resultCount) ?? results.length
  const topFamilies = results
    .slice(0, 20)
    .map(resultFamily)
    .filter((family): family is string => typeof family === "string")
  const graphOnlyEvidence =
    resultCount > 0 &&
    results.length > 0 &&
    results.every(isGraphOnlyResult) &&
    !hasAnswerGradeCandidate(diagnostics)

  return {
    questionId: question.questionId,
    questionType: question.questionType,
    searchStatus: search.status,
    evaluateStatus: evaluate.status,
    score: numberValue(evaluate.score) ?? numberValue(evaluation?.score),
    resultCount,
    searchDurationMs: numberValue(search.durationMs) ?? numberValue(evaluation?.searchDurationMs),
    scopedActivationDiagnostics: hasScopedActivationDiagnostics(diagnostics),
    graphOnlyEvidence,
    graphExpansionWithoutFrontier: graphExpansionWithoutFrontier(diagnostics),
    outOfFrontierGraphRows: outOfFrontierGraphRows(diagnostics),
    topFamilies,
  }
}

function selectedQuestionIds(
  checkpoint: RunCheckpoint,
  args: Pick<NativeActivationGateArgs, "questionIds" | "stage">
): string[] {
  const sourceIds =
    args.questionIds ?? checkpoint.targetQuestionIds ?? Object.keys(checkpoint.questions)
  if (args.stage !== "full") return sourceIds
  return sourceIds.filter(
    (questionId) => checkpoint.questions[questionId]?.questionType !== "adversarial"
  )
}

export function buildNativeActivationRunSnapshot(
  checkpoint: RunCheckpoint,
  report: BenchmarkResult | undefined,
  args: Pick<NativeActivationGateArgs, "stage" | "questionIds">
): NativeActivationRunSnapshot {
  const evaluations = evaluationByQuestion(report)
  const questionIds = selectedQuestionIds(checkpoint, args)
  const missingQuestionIds: string[] = []
  const questions: Record<string, NativeActivationQuestionSnapshot> = {}
  for (const questionId of questionIds) {
    const question = checkpoint.questions[questionId]
    if (!question) {
      missingQuestionIds.push(questionId)
      continue
    }
    questions[questionId] = questionSnapshot(question, evaluations.get(questionId))
  }

  const values = Object.values(questions)
  const searchDurations = values
    .map((question) => question.searchDurationMs)
    .filter((duration): duration is number => duration !== undefined)

  return {
    runId: checkpoint.runId,
    stage: args.stage,
    source: report ? "report+checkpoint" : "checkpoint",
    total: questionIds.length,
    searched: values.filter((question) => question.searchStatus === "completed").length,
    evaluated: values.filter((question) => question.evaluateStatus === "completed").length,
    correct: values.filter((question) => question.score === 1).length,
    failedSearches: values.filter((question) => question.searchStatus === "failed").length,
    emptySearches: values.filter(
      (question) => question.searchStatus === "completed" && question.resultCount === 0
    ).length,
    graphOnlyEvidence: values.filter((question) => question.graphOnlyEvidence).length,
    scopedDiagnosticsMissing: values.filter(
      (question) => question.searchStatus === "completed" && !question.scopedActivationDiagnostics
    ).length,
    graphExpansionWithoutFrontier: values.filter(
      (question) => question.graphExpansionWithoutFrontier
    ).length,
    outOfFrontierGraphRows: values.reduce(
      (total, question) => total + question.outOfFrontierGraphRows,
      0
    ),
    searchLatency: calculateLatencyStats(searchDurations),
    missingQuestionIds,
    questions,
  }
}

export function evaluateNativeActivationRun(
  run: NativeActivationRunSnapshot,
  args: Pick<NativeActivationGateArgs, "maxSearchP95Ms" | "expectedTotal" | "minCorrect">
): string[] {
  const failures: string[] = []
  if (run.total === 0) failures.push(`${run.runId} has no selected questions`)
  if (run.missingQuestionIds.length > 0) {
    failures.push(`${run.runId} is missing questions: ${run.missingQuestionIds.join(", ")}`)
  }
  if (args.expectedTotal !== undefined && run.total !== args.expectedTotal) {
    failures.push(`${run.runId} selected ${run.total}/${args.expectedTotal} expected questions`)
  }
  if (run.searched !== run.total)
    failures.push(`${run.runId} searched ${run.searched}/${run.total} questions`)
  if (run.evaluated !== run.total)
    failures.push(`${run.runId} evaluated ${run.evaluated}/${run.total} questions`)
  if (run.failedSearches > 0)
    failures.push(`${run.runId} has ${run.failedSearches} failed searches`)
  if (run.emptySearches > 0) failures.push(`${run.runId} has ${run.emptySearches} empty searches`)
  if (run.graphOnlyEvidence > 0) {
    failures.push(`${run.runId} has ${run.graphOnlyEvidence} graph-only evidence searches`)
  }
  if (run.scopedDiagnosticsMissing > 0) {
    failures.push(
      `${run.runId} is missing scoped activation diagnostics for ${run.scopedDiagnosticsMissing} searches`
    )
  }
  if (run.graphExpansionWithoutFrontier > 0) {
    failures.push(
      `${run.runId} expanded graph evidence without an episode frontier for ${run.graphExpansionWithoutFrontier} searches`
    )
  }
  if (!run.searchLatency) {
    failures.push(`${run.runId} search latency is unavailable`)
  } else if (run.searchLatency.p95 > args.maxSearchP95Ms) {
    failures.push(
      `${run.runId} search p95 ${run.searchLatency.p95}ms exceeds ${args.maxSearchP95Ms}ms`
    )
  }
  if (args.minCorrect !== undefined && run.correct < args.minCorrect) {
    failures.push(
      `${run.runId} correct ${run.correct}/${run.total} is below required ${args.minCorrect}`
    )
  }
  return failures
}

export function buildNativeActivationGateReport(
  run: NativeActivationRunSnapshot,
  args: Pick<NativeActivationGateArgs, "stage" | "maxSearchP95Ms" | "expectedTotal" | "minCorrect">,
  prerequisites: NativeActivationRunSnapshot[] = []
): NativeActivationGateReport {
  const failures = [
    ...prerequisites.flatMap((prerequisite) =>
      evaluateNativeActivationRun(prerequisite, {
        maxSearchP95Ms: args.maxSearchP95Ms,
        expectedTotal: undefined,
        minCorrect: undefined,
      }).map((failure) => `prerequisite ${failure}`)
    ),
    ...evaluateNativeActivationRun(run, args),
  ]
  return {
    stage: args.stage,
    run,
    prerequisites,
    failures,
  }
}

function loadSnapshot(
  runId: string,
  stage: NativeActivationGateStage,
  questionIds?: string[]
): NativeActivationRunSnapshot | null {
  const checkpoint = new CheckpointManager().load(runId)
  if (!checkpoint) return null
  return buildNativeActivationRunSnapshot(checkpoint, readReport(runId), { stage, questionIds })
}

function writeArtifact(path: string, content: string): void {
  const directory = dirname(path)
  if (directory !== ".") mkdirSync(directory, { recursive: true })
  writeFileSync(path, content)
}

function percent(correct: number, total: number): string {
  return total > 0 ? `${((correct / total) * 100).toFixed(2)}%` : "n/a"
}

function renderMarkdown(report: NativeActivationGateReport): string {
  const rows = [report.run, ...report.prerequisites]
  const lines = [
    "# MemoryKernel Native Activation Gate",
    "",
    `Stage: ${report.stage}`,
    "",
    "| Run | Correct | Searched | Empty | Graph-only | Scoped missing | Graph no frontier | Out-of-frontier rows | Search p95 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...rows.map(
      (run) =>
        `| ${run.runId} | ${run.correct}/${run.total} (${percent(run.correct, run.total)}) | ${run.searched}/${run.total} | ${run.emptySearches} | ${run.graphOnlyEvidence} | ${run.scopedDiagnosticsMissing} | ${run.graphExpansionWithoutFrontier} | ${run.outOfFrontierGraphRows} | ${run.searchLatency?.p95 ?? "n/a"} |`
    ),
    "",
    report.failures.length === 0 ? "Gate: PASS" : "Gate: FAIL",
  ]
  if (report.failures.length > 0) {
    lines.push("", "Failures:")
    for (const failure of report.failures) lines.push(`- ${failure}`)
  }
  return `${lines.join("\n")}\n`
}

function printUsage(): void {
  console.log(
    "Usage: bun run src/index.ts native-activation-gate -r <runId> --stage smoke|probe|full [options]"
  )
  console.log("")
  console.log("Options:")
  console.log("  -r, --run-id             Run identifier to gate")
  console.log("  --stage, --mode          Gate stage: smoke, probe, or full (default: full)")
  console.log(
    "  --questions              Comma/space separated question ids for smoke/probe slices"
  )
  console.log("  --max-search-p95-ms      Maximum search p95 latency (default: 6000)")
  console.log("  --expected-total         Expected selected question count (default full: 152)")
  console.log("  --min-correct            Minimum correct count (default full: 106)")
  console.log("  --smoke-run              Prerequisite smoke run for probe/full stages")
  console.log("  --probe-run              Prerequisite probe run for full stage")
  console.log("  --allow-missing-prereq   Do not fail when stage prerequisites are omitted")
  console.log("  --json                   Write JSON gate report")
  console.log("  --markdown, --md         Write Markdown gate report")
}

export async function nativeActivationGateCommand(args: string[]): Promise<void> {
  const parsed = parseNativeActivationGateArgs(args)
  if (!parsed) {
    printUsage()
    return
  }

  const run = loadSnapshot(parsed.runId, parsed.stage, parsed.questionIds)
  if (!run) {
    console.error(`No checkpoint found for run: ${parsed.runId}`)
    process.exitCode = 1
    return
  }

  const prerequisites: NativeActivationRunSnapshot[] = []
  const prerequisiteFailures: string[] = []
  if (parsed.stage === "probe" || parsed.stage === "full") {
    if (!parsed.smokeRunId && !parsed.allowMissingPrereq) {
      prerequisiteFailures.push(`${parsed.stage} stage requires --smoke-run`)
    } else if (parsed.smokeRunId) {
      const smoke = loadSnapshot(parsed.smokeRunId, "smoke")
      if (smoke) prerequisites.push(smoke)
      else
        prerequisiteFailures.push(
          `No checkpoint found for smoke prerequisite: ${parsed.smokeRunId}`
        )
    }
  }
  if (parsed.stage === "full") {
    if (!parsed.probeRunId && !parsed.allowMissingPrereq) {
      prerequisiteFailures.push("full stage requires --probe-run")
    } else if (parsed.probeRunId) {
      const probe = loadSnapshot(parsed.probeRunId, "probe")
      if (probe) prerequisites.push(probe)
      else
        prerequisiteFailures.push(
          `No checkpoint found for probe prerequisite: ${parsed.probeRunId}`
        )
    }
  }

  const report = buildNativeActivationGateReport(run, parsed, prerequisites)
  report.failures.unshift(...prerequisiteFailures)
  const markdown = renderMarkdown(report)
  console.log(markdown.trimEnd())

  if (parsed.outputJson) writeArtifact(parsed.outputJson, `${JSON.stringify(report, null, 2)}\n`)
  if (parsed.outputMarkdown) writeArtifact(parsed.outputMarkdown, markdown)

  if (report.failures.length > 0) {
    for (const failure of report.failures) console.error(`Gate failed: ${failure}`)
    process.exitCode = 1
  }
}
