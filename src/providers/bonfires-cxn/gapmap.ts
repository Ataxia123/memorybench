// Post-run gap-map classifier for bonfires-cxn LoCoMo bench runs.
//
// Consumes a completed run directory's per-question search results
// (data/runs/<run-id>/results/*.json, written by src/orchestrator/phases/search.ts)
// plus the checkpoint (judge verdicts) and the conv-26 fold artifacts (batches,
// activation log, fold plan, sidecar utterance map) and classifies every missed
// question into one of four MissClass buckets via the evidence-linkage chain:
//
//   evidence dia_id
//     -> batches message (metadata.dia_id -> timestamp, username)
//     -> activation log record ((normalized ts, speaker) match -> statement_id)
//     -> fold plan record(s) (correlation_id === statement_id, or starts with
//        `${statement_id}#`) -> utterance -> sha256(utterance)[:16] hash
//
// FIELD-PATH NOTE (pinned against real artifacts, see task-6-report.md):
// the per-question result JSON (data/runs/*/results/conv-26-qN.json) does NOT
// carry a judge verdict or evidence dia_ids — it is a search-phase-only dump
// (questionId/question/questionType/groundTruth/containerTag/timestamp/
// durationMs/diagnostics/results). The judge verdict lives in the run's
// checkpoint.json at questions[id].phases.evaluate.label ("correct"|
// "incorrect"), and evidence dia_ids are never persisted anywhere — they only
// exist in the raw LoCoMo dataset (data/benchmarks/locomo/locomo10.json,
// qa[i].evidence), loaded in-memory by src/benchmarks/locomo/index.ts. The
// CLI (scripts/cxn-gapmap.ts) is responsible for supplying both via
// `GapRunPaths.evidenceByQuestionId` / `.questionTextById` — classifyRun/
// loadGapInputs stay decoupled from the benchmark loader so they're testable
// with plain fixtures.
//
// INCOMPLETE-RUN NOTE: a checkpoint question can be missing a completed
// evaluate label (evaluate phase never ran) or missing its
// results/<id>.json entirely (search phase never persisted). Neither state
// is a judge verdict, so loadGapInputs excludes both from `questions`
// (and thus from the 4-class taxonomy) and surfaces them instead as
// `GapInputs.incompleteQuestions` / `GapReport.incomplete` — see
// IncompleteQuestion below. A partial run is a valid gap map, just loudly
// annotated rather than silently minting phantom misses or dropping
// questions from `total`.

import { readFile, readdir } from "node:fs/promises"
import { join } from "node:path"
import { createHash } from "node:crypto"
import { extractTerms, STOPWORDS } from "./retrieval"

export type MissClass = "not_hydratable" | "not_retrieved" | "answered_wrong" | "unlinked"

// A checkpoint question that never made it into the 4-class taxonomy because
// the run itself is incomplete (see loadGapInputs) — NOT a judge verdict of
// any kind, so it must not be silently folded into `overall.total`/misses.
export interface IncompleteQuestion {
  questionId: string
  // "unevaluated": results/<id>.json exists (search ran) but
  // checkpoint.json's questions[id].phases.evaluate has no label (evaluate
  // phase never completed, e.g. still "pending").
  // "missingResults": checkpoint.json has a completed evaluate label for
  // this id, but results/<id>.json is absent from disk.
  reason: "unevaluated" | "missingResults"
}

export interface GapReport {
  overall: { total: number; correct: number; score: number }
  byCategory: Record<string, { total: number; correct: number; missByClass: Record<MissClass, number> }>
  // Questions excluded from `overall`/`byCategory` because the run is
  // incomplete for them (I2) — counts only; see `incompleteIds` for the ids.
  incomplete: { unevaluated: number; missingResults: number }
  incompleteIds: { unevaluated: string[]; missingResults: string[] }
  // Of the misses classified `unlinked`, how many had an empty evidence
  // array to begin with (vacuously unlinked, no dia_id to report as failed —
  // M3). A sub-count of byCategory[*].missByClass.unlinked, not a new class.
  emptyEvidenceUnlinkedCount: number
  // Hit@20, decoupled from the judge verdict (leg 2): a question "hits" iff
  // ANY of its linked evidence hashes appears in its retrieved
  // metadata.utterance_hash set. Computed over EVERY scored question in
  // `inputs.questions` — correct ones included — because this measures pool
  // quality (did the evidence even land in the retrieved set), not answer
  // conversion. Deliberately independent of `correct`/`missByClass`: a
  // correct question can still miss its evidence in the top-k (the judge may
  // have been satisfied by something else), and a miss can still hit (the
  // model saw the evidence but answered wrong — see `answered_wrong`).
  hitAt20: {
    overall: { hits: number; total: number }
    byCategory: Record<string, { hits: number; total: number }>
  }
  junkSeedReport: Array<{ name: string; questionCount: number }>
  unlinkedEvidence: string[]
  // v3: comprehend-sidecar affordance channels (see src/providers/bonfires-cxn/
  // index.ts's search(), recipe.affordancesFired/recipe.fallback, only present
  // when the run had CXN_Q=1). Rates are over SCORED questions (the same
  // denominator as `overall.total` — incomplete questions excluded). A run
  // with no v3 recipe metadata at all (leg-1/leg-2 runs, or a v3 run re-run
  // with CXN_Q=0) yields all-zero rates here rather than crashing — see
  // computeAffordances / extractAffordances.
  affordances: AffordancesReport
  // v3: only present when the CLI was invoked with --control-dir. Compares
  // this run's ("arm") per-question correctness against a second run's
  // ("control") — see computeFlips.
  flips?: FlipsReport
}

// Affordance keys the comprehend sidecar can fire, per src/providers/
// bonfires-cxn/index.ts's search() (`affordancesFired.push("q:...")`) — kept
// as a literal tuple (not derived from recipe) so a run with zero v3
// metadata still reports all four keys at rate 0 rather than an empty object.
export const AFFORDANCE_KEYS = ["q:strata", "q:temporal", "q:seed", "q:answer"] as const
export type AffordanceKey = (typeof AFFORDANCE_KEYS)[number]

export interface AffordancesReport {
  fallbackRate: number
  fallbackByCategory: Record<string, number>
  fireRates: Record<string, number>
}

export interface FlipsReport {
  gained: string[]
  lost: string[]
  gainedByCategory: Record<string, number>
  lostByCategory: Record<string, number>
}

export interface GapQuestionResult {
  questionId: string
  questionType: string
  correct: boolean
  evidence: string[]
  retrievedHashes: string[]
  // v3: recipe.affordancesFired / recipe.fallback from the run's persisted
  // cxn_context search item (see extractAffordances). Empty array / false
  // when the run has no v3 recipe metadata at all — never absent, so
  // computeAffordances never has to special-case v2 shape.
  affordancesFired: string[]
  fallback: boolean
}

// Minimal shape computeFlips needs — GapQuestionResult satisfies it
// structurally, and loadControlQuestions returns exactly this (no evidence/
// retrievedHashes needed for a correctness-only comparison run).
export interface CorrectnessRecord {
  questionId: string
  questionType: string
  correct: boolean
}

export interface BatchMessage {
  username: string
  timestamp: string
  metadata: { dia_id: string }
}

export interface ActivationLogRecord {
  statement_id: string
  speaker: string
  ts: string
}

export interface FoldPlanRecord {
  correlation_id: string
  utterance: string
  event_ts: string
  actor_id: string
}

export interface UtteranceMapEntry {
  utterance: string
  ts: string
  actor_id: string
  session: string
}

export interface GapInputs {
  questions: GapQuestionResult[]
  batchMessages: BatchMessage[]
  logRecords: ActivationLogRecord[]
  planRecords: FoldPlanRecord[]
  utteranceMap: Record<string, UtteranceMapEntry>
  // Scoped to just the questionIds that appear in THIS run (union of
  // results/*.json ids and checkpoint.json question ids) — NOT the full
  // LoCoMo corpus (M4). See loadGapInputs.
  allQuestionTexts: string[]
  incompleteQuestions: IncompleteQuestion[]
}

// conv-26's activation-log/batches timestamps are UTC-only, so "+00:00" is
// the only offset form ever produced — this normalization would silently
// fail to link evidence (falling through to "unlinked") for any other
// timezone offset (M5).
function normalizeTs(ts: string): string {
  return ts.replace("+00:00", "Z")
}

function utteranceHash(utterance: string): string {
  return createHash("sha256").update(utterance, "utf-8").digest("hex").slice(0, 16)
}

interface LinkageResult {
  linkedDiaIds: string[]
  unlinkedDiaIds: string[]
  hashes: Set<string>
}

// N1: LoCoMo occasionally joins multiple dia_ids into a single evidence array
// element, e.g. "D8:6; D9:17" — one string, two references. Splitting them
// out here (the evidence-iteration entry point, shared by every caller of
// linkEvidence) means both the miss-classification chain and Hit@20 see the
// same, correctly-split dia_id list; unsplit, the joined string can never
// match a batches message's dia_id and silently counts as one unresolvable
// evidence reference instead of two resolvable ones.
function splitEvidenceIds(evidence: string[]): string[] {
  return evidence.flatMap((entry) =>
    entry
      .split(/;\s*/)
      .map((id) => id.trim())
      .filter((id) => id.length > 0)
  )
}

function linkEvidence(
  evidence: string[],
  batchMessages: BatchMessage[],
  logRecords: ActivationLogRecord[],
  planRecords: FoldPlanRecord[]
): LinkageResult {
  const messageByDiaId = new Map<string, BatchMessage>()
  for (const m of batchMessages) {
    if (m.metadata?.dia_id && !messageByDiaId.has(m.metadata.dia_id)) {
      messageByDiaId.set(m.metadata.dia_id, m)
    }
  }

  const statementIdsByKey = new Map<string, string[]>()
  for (const rec of logRecords) {
    const key = `${normalizeTs(rec.ts)}\x00${rec.speaker}`
    const arr = statementIdsByKey.get(key)
    if (arr) arr.push(rec.statement_id)
    else statementIdsByKey.set(key, [rec.statement_id])
  }

  // correlation_id is either the bare statement_id (a "card" record) or
  // `${statement_id}#<suffix>` (a "residual" record) — extracting everything
  // before the first "#" is equivalent to the brief's
  // `correlation_id.startsWith(statement_id + "#")` rule but lets us index
  // once instead of scanning the whole plan per statement_id.
  const planByStatementId = new Map<string, FoldPlanRecord[]>()
  for (const rec of planRecords) {
    const base = rec.correlation_id.split("#")[0]!
    const arr = planByStatementId.get(base)
    if (arr) arr.push(rec)
    else planByStatementId.set(base, [rec])
  }

  const linkedDiaIds: string[] = []
  const unlinkedDiaIds: string[] = []
  const hashes = new Set<string>()

  for (const diaId of splitEvidenceIds(evidence)) {
    const message = messageByDiaId.get(diaId)
    if (!message) {
      unlinkedDiaIds.push(diaId)
      continue
    }
    const key = `${normalizeTs(message.timestamp)}\x00${message.username}`
    const statementIds = statementIdsByKey.get(key)
    if (!statementIds || statementIds.length === 0) {
      unlinkedDiaIds.push(diaId)
      continue
    }
    // Linkage succeeds the moment we resolve a statement_id — whether that
    // statement produced any plan record (and thus a hash) is a SEPARATE
    // concern (not_hydratable), not part of "unlinked".
    linkedDiaIds.push(diaId)
    for (const statementId of statementIds) {
      const records = planByStatementId.get(statementId) ?? []
      for (const record of records) hashes.add(utteranceHash(record.utterance))
    }
  }

  return { linkedDiaIds, unlinkedDiaIds, hashes }
}

// Miss classification from an already-computed linkage (see classifyRun,
// which now computes linkage once per question and shares it with Hit@20 —
// NOTE: an empty `evidence` array yields the same empty LinkageResult as a
// non-empty-but-fully-unresolvable one, so the "no evidence at all" case
// doesn't need a special early return here; both fall out of
// `linkedDiaIds.length === 0` below identically to before this refactor.
function classifyMissClass(
  linkage: LinkageResult,
  retrievedHashes: string[],
  utteranceMap: Record<string, UtteranceMapEntry>
): MissClass {
  if (linkage.linkedDiaIds.length === 0) {
    return "unlinked"
  }

  const hydratedHashes = [...linkage.hashes].filter((h) => Object.prototype.hasOwnProperty.call(utteranceMap, h))
  if (hydratedHashes.length === 0) {
    return "not_hydratable"
  }

  const retrieved = new Set(retrievedHashes)
  const anyRetrieved = hydratedHashes.some((h) => retrieved.has(h))
  if (!anyRetrieved) {
    return "not_retrieved"
  }

  return "answered_wrong"
}

const EMPTY_MISS_BY_CLASS: Record<MissClass, number> = {
  not_hydratable: 0,
  not_retrieved: 0,
  answered_wrong: 0,
  unlinked: 0,
}

// Rates are over SCORED questions — same population as `overall.total`
// (questions.length here; incomplete questions never reach GapQuestionResult
// at all, see loadGapInputs). A zero-length `questions` (e.g. wholly
// incomplete run) yields all-zero rates rather than NaN.
export function computeAffordances(questions: GapQuestionResult[]): AffordancesReport {
  const total = questions.length

  const fireCounts: Record<string, number> = {}
  for (const key of AFFORDANCE_KEYS) fireCounts[key] = 0

  let fallbackCount = 0
  const totalByCategory = new Map<string, number>()
  const fallbackCountByCategory = new Map<string, number>()

  for (const q of questions) {
    totalByCategory.set(q.questionType, (totalByCategory.get(q.questionType) ?? 0) + 1)
    for (const key of q.affordancesFired) {
      if (Object.prototype.hasOwnProperty.call(fireCounts, key)) fireCounts[key] += 1
    }
    if (q.fallback) {
      fallbackCount += 1
      fallbackCountByCategory.set(q.questionType, (fallbackCountByCategory.get(q.questionType) ?? 0) + 1)
    }
  }

  const fireRates: Record<string, number> = {}
  for (const key of AFFORDANCE_KEYS) fireRates[key] = total > 0 ? fireCounts[key]! / total : 0

  const fallbackByCategory: Record<string, number> = {}
  for (const category of [...totalByCategory.keys()].sort()) {
    const categoryTotal = totalByCategory.get(category)!
    fallbackByCategory[category] = categoryTotal > 0 ? (fallbackCountByCategory.get(category) ?? 0) / categoryTotal : 0
  }

  return {
    fallbackRate: total > 0 ? fallbackCount / total : 0,
    fallbackByCategory,
    fireRates,
  }
}

// Question ids "correct in arm but not control" (gained) and "correct in
// control but not arm" (lost) — a per-question-id set comparison, not scoped
// to ids present in both runs (a control run that never scored an id the arm
// scored correctly still counts as a gain: the arm produced a correct answer
// the control run didn't). *ByCategory buckets use the reporting side's own
// questionType (arm's for gained, control's for lost) since that's the run
// the id's category is actually being attributed to.
export function computeFlips(arm: CorrectnessRecord[], control: CorrectnessRecord[]): FlipsReport {
  const armById = new Map(arm.map((q) => [q.questionId, q]))
  const controlById = new Map(control.map((q) => [q.questionId, q]))
  const armCorrect = new Set(arm.filter((q) => q.correct).map((q) => q.questionId))
  const controlCorrect = new Set(control.filter((q) => q.correct).map((q) => q.questionId))

  const gained = [...armCorrect].filter((id) => !controlCorrect.has(id)).sort()
  const lost = [...controlCorrect].filter((id) => !armCorrect.has(id)).sort()

  const gainedByCategory: Record<string, number> = {}
  for (const id of gained) {
    const category = armById.get(id)?.questionType ?? "unknown"
    gainedByCategory[category] = (gainedByCategory[category] ?? 0) + 1
  }
  const lostByCategory: Record<string, number> = {}
  for (const id of lost) {
    const category = controlById.get(id)?.questionType ?? "unknown"
    lostByCategory[category] = (lostByCategory[category] ?? 0) + 1
  }

  const sortRecord = (rec: Record<string, number>): Record<string, number> => {
    const out: Record<string, number> = {}
    for (const key of Object.keys(rec).sort()) out[key] = rec[key]!
    return out
  }

  return { gained, lost, gainedByCategory: sortRecord(gainedByCategory), lostByCategory: sortRecord(lostByCategory) }
}

export function classifyRun(inputs: GapInputs): GapReport {
  const total = inputs.questions.length
  const correct = inputs.questions.filter((q) => q.correct).length
  const overall = { total, correct, score: total > 0 ? correct / total : 0 }

  const byCategory: Record<string, { total: number; correct: number; missByClass: Record<MissClass, number> }> = {}
  const hitAt20ByCategory: Record<string, { hits: number; total: number }> = {}
  const hitAt20Overall = { hits: 0, total: 0 }
  const unlinkedSet = new Set<string>()
  let emptyEvidenceUnlinkedCount = 0

  const sortedQuestions = [...inputs.questions].sort((a, b) =>
    a.questionId < b.questionId ? -1 : a.questionId > b.questionId ? 1 : 0
  )

  for (const question of sortedQuestions) {
    const category = (byCategory[question.questionType] ??= {
      total: 0,
      correct: 0,
      missByClass: { ...EMPTY_MISS_BY_CLASS },
    })
    const hitCategory = (hitAt20ByCategory[question.questionType] ??= { hits: 0, total: 0 })
    category.total += 1

    // Linkage is computed once per question (evidence dia_id -> hashes) and
    // shared by both the 4-class miss taxonomy (correct questions skip it,
    // see the `continue` below) and Hit@20 (which runs over EVERY scored
    // question, correct included — Hit@20 measures whether the evidence
    // landed in the retrieved pool at all, a strictly weaker and separate
    // question from whether the judge was satisfied).
    const linkage = linkEvidence(question.evidence, inputs.batchMessages, inputs.logRecords, inputs.planRecords)
    const retrievedSet = new Set(question.retrievedHashes)
    const isHit = [...linkage.hashes].some((h) => retrievedSet.has(h))
    hitAt20Overall.total += 1
    hitCategory.total += 1
    if (isHit) {
      hitAt20Overall.hits += 1
      hitCategory.hits += 1
    }

    if (question.correct) {
      category.correct += 1
      continue
    }

    const cls = classifyMissClass(linkage, question.retrievedHashes, inputs.utteranceMap)
    category.missByClass[cls] += 1
    if (cls === "unlinked" && question.evidence.length === 0) emptyEvidenceUnlinkedCount += 1
    // unlinkedEvidence stays scoped to MISS-linkage failures only (not every
    // question's linkage, even though linkage above now also runs for
    // correct questions for Hit@20) — this loop only reaches here for
    // question.correct === false, so a correct question's unresolvable
    // dia_ids (if any) never enter unlinkedSet. Kept this way deliberately:
    // unlinkedEvidence is a debugging aid for the miss taxonomy ("why didn't
    // this dia_id classify"), not a general evidence-linkage health report;
    // widening its scope would be a separate, explicit change.
    for (const diaId of linkage.unlinkedDiaIds) unlinkedSet.add(diaId)
  }

  const sortedByCategory: typeof byCategory = {}
  for (const key of Object.keys(byCategory).sort()) sortedByCategory[key] = byCategory[key]!

  const sortedHitAt20ByCategory: typeof hitAt20ByCategory = {}
  for (const key of Object.keys(hitAt20ByCategory).sort()) sortedHitAt20ByCategory[key] = hitAt20ByCategory[key]!

  const unevaluatedIds = inputs.incompleteQuestions
    .filter((q) => q.reason === "unevaluated")
    .map((q) => q.questionId)
    .sort()
  const missingResultsIds = inputs.incompleteQuestions
    .filter((q) => q.reason === "missingResults")
    .map((q) => q.questionId)
    .sort()

  return {
    overall,
    byCategory: sortedByCategory,
    incomplete: { unevaluated: unevaluatedIds.length, missingResults: missingResultsIds.length },
    incompleteIds: { unevaluated: unevaluatedIds, missingResults: missingResultsIds },
    emptyEvidenceUnlinkedCount,
    hitAt20: { overall: hitAt20Overall, byCategory: sortedHitAt20ByCategory },
    junkSeedReport: computeJunkSeedReport(inputs.allQuestionTexts),
    unlinkedEvidence: [...unlinkedSet].sort(),
    affordances: computeAffordances(inputs.questions),
  }
}

function junkCandidateNames(): string[] {
  return [...new Set<string>(["who", "true", "that", ...STOPWORDS])]
}

function topTen(
  counts: Array<{ name: string; questionCount: number }>
): Array<{ name: string; questionCount: number }> {
  return [...counts]
    .sort((a, b) => b.questionCount - a.questionCount || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    .slice(0, 10)
}

// Literal spec: re-run extractTerms (which strips STOPWORDS) over every
// question and count exact-token matches against the candidate junk names.
// FINDING: because "who"/"that" (and every other candidate besides "true")
// are themselves in STOPWORDS, extractTerms strips them before the
// exact-token check ever runs, so this report is degenerate — always 0 for
// every stopword-named candidate. That's good news, not a bug: it means the
// provider's term extraction already shields against stopword-named junk KG
// entities. See computeRawTokenJunkCounts for the non-degenerate remainder.
export function computeJunkSeedReport(
  questionTexts: string[]
): Array<{ name: string; questionCount: number }> {
  const candidates = junkCandidateNames()
  const termLists = questionTexts.map((q) => new Set(extractTerms(q)))
  const counts = candidates.map((name) => ({
    name,
    questionCount: termLists.reduce((n, terms) => n + (terms.has(name) ? 1 : 0), 0),
  }))
  return topTen(counts)
}

// Non-degenerate remainder: same candidate names, counted by raw lowercase
// token presence in the question text BEFORE stopword filtering. This shows
// what the junk-seed counts would look like if a junk KG entity were named
// "who"/"that"/etc without extractTerms's STOPWORDS shield in front of it.
// Diagnostic only — NOT part of GapReport's shape.
export function computeRawTokenJunkCounts(
  questionTexts: string[]
): Array<{ name: string; questionCount: number }> {
  const candidates = junkCandidateNames()
  const tokenSets = questionTexts.map(
    (q) => new Set(q.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((t) => t.length > 0))
  )
  const counts = candidates.map((name) => ({
    name,
    questionCount: tokenSets.reduce((n, tokens) => n + (tokens.has(name) ? 1 : 0), 0),
  }))
  return topTen(counts)
}

function parseJsonl<T>(text: string): T[] {
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as T)
}

function normalizeBatchMessage(raw: unknown): BatchMessage {
  const m = (raw ?? {}) as Record<string, unknown>
  const metadata = (m.metadata ?? {}) as Record<string, unknown>
  const username = m.username ?? m.speaker ?? ""
  return {
    username: String(username ?? ""),
    timestamp: String(m.timestamp ?? ""),
    metadata: { dia_id: String(metadata.dia_id ?? "") },
  }
}

function flattenBatches(raw: unknown): BatchMessage[] {
  if (!Array.isArray(raw)) return []
  const out: BatchMessage[] = []
  for (const item of raw) {
    if (Array.isArray(item)) {
      for (const message of item) out.push(normalizeBatchMessage(message))
    } else {
      out.push(normalizeBatchMessage(item))
    }
  }
  return out
}

// v3: pulls recipe.affordancesFired / recipe.fallback off the persisted
// cxn_context search item (see src/providers/bonfires-cxn/index.ts's
// search() — the last item in the returned array, kind === "cxn_context").
// A v2 run (or a v3 run with CXN_Q=0, where recipe is baseRecipe with no q
// fields at all) has no such fields on the recipe — or no cxn_context item
// at all — and this falls through to the same {[], false} default either
// way, which is exactly the "zero rates, no crash" requirement.
function extractAffordances(raw: {
  results?: Array<{ kind?: unknown; recipe?: unknown }>
}): { affordancesFired: string[]; fallback: boolean } {
  for (const item of raw.results ?? []) {
    if (!item || typeof item !== "object" || item.kind !== "cxn_context") continue
    const recipe = (item as { recipe?: unknown }).recipe
    if (!recipe || typeof recipe !== "object") return { affordancesFired: [], fallback: false }
    const recipeRecord = recipe as Record<string, unknown>
    const affordancesFired = Array.isArray(recipeRecord.affordancesFired)
      ? recipeRecord.affordancesFired.filter((v): v is string => typeof v === "string")
      : []
    const fallback = typeof recipeRecord.fallback === "boolean" ? recipeRecord.fallback : false
    return { affordancesFired, fallback }
  }
  return { affordancesFired: [], fallback: false }
}

export interface GapRunPaths {
  runDir: string
  batchesPath: string
  logPath: string
  planPath: string
  mapPath: string
  // Sourced by the CLI from the LoCoMo dataset (never persisted in run
  // artifacts) — see the FIELD-PATH NOTE at the top of this file. Keyed by
  // questionId so loadGapInputs can scope the corpus down to just this run's
  // questions (M4) instead of the full ~1986-question LoCoMo set.
  evidenceByQuestionId: Record<string, string[]>
  questionTextById: Record<string, string>
}

export async function loadGapInputs(paths: GapRunPaths): Promise<GapInputs> {
  const checkpointRaw = JSON.parse(await readFile(join(paths.runDir, "checkpoint.json"), "utf-8")) as {
    questions: Record<string, { phases?: { evaluate?: { label?: string } } }>
  }

  const resultsDir = join(paths.runDir, "results")
  const resultFiles = (await readdir(resultsDir)).filter((f) => f.endsWith(".json")).sort()
  const resultQuestionIds = new Set(resultFiles.map((f) => f.replace(/\.json$/, "")))

  const questions: GapQuestionResult[] = []
  const incompleteQuestions: IncompleteQuestion[] = []
  const runQuestionIds = new Set<string>()

  for (const file of resultFiles) {
    const raw = JSON.parse(await readFile(join(resultsDir, file), "utf-8")) as {
      questionId: string
      questionType: string
      results?: Array<{ kind?: unknown; recipe?: unknown; metadata?: { utterance_hash?: string } }>
    }
    runQuestionIds.add(raw.questionId)
    const label = checkpointRaw.questions[raw.questionId]?.phases?.evaluate?.label
    if (typeof label !== "string") {
      // Evaluate phase never completed for this question (e.g. checkpoint
      // status "pending", or no checkpoint entry at all) — exclude from the
      // 4-class taxonomy entirely rather than silently minting a phantom
      // miss (I2a).
      incompleteQuestions.push({ questionId: raw.questionId, reason: "unevaluated" })
      continue
    }
    const retrievedHashes = (raw.results ?? [])
      .map((r) => r.metadata?.utterance_hash)
      .filter((h): h is string => typeof h === "string")
    const { affordancesFired, fallback } = extractAffordances(raw)
    questions.push({
      questionId: raw.questionId,
      questionType: raw.questionType,
      correct: label === "correct",
      evidence: paths.evidenceByQuestionId[raw.questionId] ?? [],
      retrievedHashes,
      affordancesFired,
      fallback,
    })
  }

  // A checkpoint question with a completed evaluate label but no
  // results/<id>.json on disk must not silently vanish from `total` (I2b) —
  // count it as incomplete instead of dropping it.
  for (const questionId of Object.keys(checkpointRaw.questions)) {
    runQuestionIds.add(questionId)
    if (resultQuestionIds.has(questionId)) continue
    incompleteQuestions.push({ questionId, reason: "missingResults" })
  }

  const batchesRaw = JSON.parse(await readFile(paths.batchesPath, "utf-8")) as unknown
  const batchMessages = flattenBatches(batchesRaw)

  const logRecords = parseJsonl<ActivationLogRecord>(await readFile(paths.logPath, "utf-8"))
  const planRecords = parseJsonl<FoldPlanRecord>(await readFile(paths.planPath, "utf-8"))
  const utteranceMap = JSON.parse(await readFile(paths.mapPath, "utf-8")) as Record<string, UtteranceMapEntry>

  const allQuestionTexts = [...runQuestionIds]
    .sort()
    .map((id) => paths.questionTextById[id])
    .filter((text): text is string => typeof text === "string")

  return {
    questions,
    batchMessages,
    logRecords,
    planRecords,
    utteranceMap,
    allQuestionTexts,
    incompleteQuestions,
  }
}

// v3 --control-dir support: a control run only needs to answer "was this
// question id correct, and what category is it" to feed computeFlips — it
// never needs evidence linkage (batches/log/plan/map), so this is a
// deliberately lighter loader than loadGapInputs, not a re-use of it.
// Incomplete questions (no evaluate label, e.g. still "pending") are
// excluded the same way loadGapInputs excludes them from `questions` — a
// question the control run never scored can't be "correct in control" for
// the flips comparison.
export async function loadControlQuestions(runDir: string): Promise<CorrectnessRecord[]> {
  const checkpointRaw = JSON.parse(await readFile(join(runDir, "checkpoint.json"), "utf-8")) as {
    questions: Record<string, { phases?: { evaluate?: { label?: string } } }>
  }
  const resultsDir = join(runDir, "results")
  const resultFiles = (await readdir(resultsDir)).filter((f) => f.endsWith(".json")).sort()

  const out: CorrectnessRecord[] = []
  for (const file of resultFiles) {
    const raw = JSON.parse(await readFile(join(resultsDir, file), "utf-8")) as {
      questionId: string
      questionType: string
    }
    const label = checkpointRaw.questions[raw.questionId]?.phases?.evaluate?.label
    if (typeof label !== "string") continue
    out.push({ questionId: raw.questionId, questionType: raw.questionType, correct: label === "correct" })
  }
  return out
}
