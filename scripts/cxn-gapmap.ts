#!/usr/bin/env bun
// Post-run gap-map CLI: classifies every miss in a bonfires-cxn LoCoMo run
// into not_hydratable / not_retrieved / answered_wrong / unlinked, plus a
// junk-seed report. See src/providers/bonfires-cxn/gapmap.ts for the field-
// path pinning notes and linkage algorithm.
//
// Usage:
//   bun run scripts/cxn-gapmap.ts \
//     --run-dir data/runs/<run-id> \
//     --batches <conv26_batches.json> \
//     --log <activation_log.jsonl> \
//     --plan <fold_plan.jsonl> \
//     --map <utterance_map.json> \
//     --out <report-dir>

import { writeFileSync, mkdirSync, existsSync } from "node:fs"
import { join } from "node:path"
import {
  classifyRun,
  loadGapInputs,
  computeRawTokenJunkCounts,
  type GapReport,
} from "../src/providers/bonfires-cxn/gapmap"
import { LoCoMoBenchmark } from "../src/benchmarks/locomo/index"

const REQUIRED_FLAGS = ["run-dir", "batches", "log", "plan", "map", "out"] as const

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!
    if (!arg.startsWith("--")) continue
    const key = arg.slice(2)
    const value = argv[i + 1]
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`--${key} requires a value`)
    }
    out[key] = value
    i++
  }
  return out
}

// JSON.stringify preserves insertion order for string keys; sort recursively
// so gap-report.json is deterministic regardless of how the report was built.
function sortKeysDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeysDeep)
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      out[key] = sortKeysDeep((value as Record<string, unknown>)[key])
    }
    return out
  }
  return value
}

function renderMarkdown(
  report: GapReport,
  rawRemainder: Array<{ name: string; questionCount: number }>
): string {
  const lines: string[] = []
  lines.push("# Gap-map report", "")
  lines.push("## Overall", "")
  lines.push(
    "| total (scored) | correct | score | incomplete: unevaluated | incomplete: missing results |",
    "|---|---|---|---|---|"
  )
  lines.push(
    `| ${report.overall.total} | ${report.overall.correct} | ${report.overall.score.toFixed(4)} | ${report.incomplete.unevaluated} | ${report.incomplete.missingResults} |`,
    ""
  )
  lines.push(
    "`total` counts only questions that entered scoring (correct + classified misses). Incomplete questions are excluded from `total` and the taxonomy below — see the Incomplete section.",
    ""
  )

  lines.push("## Incomplete (excluded from taxonomy)", "")
  lines.push(
    `${report.incomplete.unevaluated} question(s) never completed the evaluate phase (no judge label yet) and ${report.incomplete.missingResults} question(s) have a completed judge label but no \`results/<id>.json\` on disk. Neither is a "correct" or a classified miss — both are incomplete-run artifacts, listed here instead of silently dropped or counted as phantom misses.`,
    ""
  )
  lines.push("### Unevaluated (no evaluate label)", "")
  if (report.incompleteIds.unevaluated.length === 0) {
    lines.push("- none", "")
  } else {
    for (const id of report.incompleteIds.unevaluated) lines.push(`- ${id}`)
    lines.push("")
  }
  lines.push("### Missing results file", "")
  if (report.incompleteIds.missingResults.length === 0) {
    lines.push("- none", "")
  } else {
    for (const id of report.incompleteIds.missingResults) lines.push(`- ${id}`)
    lines.push("")
  }

  lines.push("## By category", "")
  lines.push(
    "| category | total | correct | not_hydratable | not_retrieved | answered_wrong | unlinked |",
    "|---|---|---|---|---|---|---|"
  )
  for (const key of Object.keys(report.byCategory).sort()) {
    const c = report.byCategory[key]!
    lines.push(
      `| ${key} | ${c.total} | ${c.correct} | ${c.missByClass.not_hydratable} | ${c.missByClass.not_retrieved} | ${c.missByClass.answered_wrong} | ${c.missByClass.unlinked} |`
    )
  }
  lines.push("")

  lines.push("## Junk-seed report (top 10, literal spec via extractTerms, scoped to this run's questions)", "")
  lines.push(
    'STOPWORDS-filtered — names that are themselves stopwords (e.g. "who", "that") can never surface here because extractTerms strips them before the exact-token check runs. That degeneracy is itself the finding: the provider\'s term extraction already shields against stopword-named junk KG entities. Computed over only the questionIds that appear in this run, not the full LoCoMo corpus.',
    ""
  )
  lines.push("| name | questionCount |", "|---|---|")
  for (const row of report.junkSeedReport) lines.push(`| ${row.name} | ${row.questionCount} |`)
  lines.push("")

  lines.push("## Non-degenerate remainder (raw token presence — diagnostic only, NOT part of GapReport)", "")
  lines.push(
    "Same candidate names, counted by raw lowercase token presence in the question text before stopword filtering. Shows what the junk-seed counts would look like without extractTerms's STOPWORDS shield. Also scoped to this run's questions.",
    ""
  )
  lines.push("| name | questionCount |", "|---|---|")
  for (const row of rawRemainder) lines.push(`| ${row.name} | ${row.questionCount} |`)
  lines.push("")

  lines.push(`## Unlinked evidence dia_ids (${report.unlinkedEvidence.length})`, "")
  lines.push(
    `Of the misses classified \`unlinked\`, ${report.emptyEvidenceUnlinkedCount} had an empty evidence array to begin with (vacuously unlinked — no dia_id to report as failed). The list below covers individually-unlinked dia_ids from the remaining unlinked misses.`,
    ""
  )
  for (const diaId of report.unlinkedEvidence) lines.push(`- ${diaId}`)
  lines.push("")

  return lines.join("\n")
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const missing = REQUIRED_FLAGS.filter((flag) => !args[flag])
  if (missing.length > 0) {
    console.error(`missing required flag(s): ${missing.map((f) => `--${f}`).join(", ")}`)
    console.error(
      "usage: bun run scripts/cxn-gapmap.ts --run-dir <dir> --batches <file> --log <file> --plan <file> --map <file> --out <dir>"
    )
    process.exit(1)
  }

  // Evidence dia_ids are never persisted in run artifacts (see gapmap.ts's
  // FIELD-PATH NOTE) — they only exist in the raw LoCoMo dataset, loaded here
  // via the same benchmark class the orchestrator uses.
  const benchmark = new LoCoMoBenchmark()
  await benchmark.load()
  const allQuestions = benchmark.getQuestions()
  const evidenceByQuestionId: Record<string, string[]> = {}
  const questionTextById: Record<string, string> = {}
  for (const q of allQuestions) {
    const evidence = (q.metadata as { evidence?: unknown } | undefined)?.evidence
    evidenceByQuestionId[q.questionId] = Array.isArray(evidence) ? evidence.map(String) : []
    questionTextById[q.questionId] = q.question
  }

  const inputs = await loadGapInputs({
    runDir: args["run-dir"]!,
    batchesPath: args.batches!,
    logPath: args.log!,
    planPath: args.plan!,
    mapPath: args.map!,
    evidenceByQuestionId,
    questionTextById,
  })

  const report = classifyRun(inputs)
  // Scoped to this run's questionIds (M4), not the full ~1986-question
  // LoCoMo corpus — loadGapInputs already filtered allQuestionTexts down.
  const rawRemainder = computeRawTokenJunkCounts(inputs.allQuestionTexts)

  const outDir = args.out!
  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true })
  writeFileSync(join(outDir, "gap-report.json"), JSON.stringify(sortKeysDeep(report), null, 2) + "\n")
  writeFileSync(join(outDir, "gap-report.md"), renderMarkdown(report, rawRemainder))
  console.log(
    `wrote ${join(outDir, "gap-report.json")} and ${join(outDir, "gap-report.md")} (${report.overall.total} questions, ${report.overall.correct} correct)`
  )

  const incompleteTotal = report.incomplete.unevaluated + report.incomplete.missingResults
  if (incompleteTotal > 0) {
    console.error(
      `WARNING: run is incomplete — ${incompleteTotal} question(s) excluded from the gap-map taxonomy ` +
        `(unevaluated=${report.incomplete.unevaluated}, missingResults=${report.incomplete.missingResults}). ` +
        `See gap-report.md's "Incomplete (excluded from taxonomy)" section. This is not a failure; the ` +
        `gap map for the completed questions is still valid.`
    )
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
