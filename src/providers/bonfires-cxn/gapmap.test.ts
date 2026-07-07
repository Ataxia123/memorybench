import { describe, expect, test } from "bun:test"
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { createHash } from "node:crypto"
import {
  classifyRun,
  loadGapInputs,
  computeJunkSeedReport,
  computeRawTokenJunkCounts,
  type GapReport,
  type GapRunPaths,
} from "./gapmap"

// Mirrors memory_kernel/scripts/eval/emit_utterance_map.py's utterance_hash():
// sha256(utterance)[:16]. Duplicated here (not imported) so the fixture
// builds its sidecar map the same way a real run would, independently of
// gapmap.ts's internal hashing.
function utteranceHash(utterance: string): string {
  return createHash("sha256").update(utterance, "utf-8").digest("hex").slice(0, 16)
}

const H1 = utteranceHash("Caroline greeted Mel.")
const H2 = utteranceHash("Melanie felt supported.")

type Fixture = GapRunPaths

const BOGUS_QUESTION_TEXT = "This text must never appear in the run-scoped junk corpus."

function buildFixture(): Fixture {
  const root = mkdtempSync(join(tmpdir(), "cxn-gapmap-"))
  const runDir = join(root, "run")
  const resultsDir = join(runDir, "results")
  mkdirSync(resultsDir, { recursive: true })

  // --- checkpoint.json: only questions[id].phases.evaluate.label is read ---
  writeFileSync(
    join(runDir, "checkpoint.json"),
    JSON.stringify({
      questions: {
        "conv-26-q1": { phases: { evaluate: { label: "correct" } } },
        "conv-26-q2": { phases: { evaluate: { label: "incorrect" } } },
        "conv-26-q3": { phases: { evaluate: { label: "incorrect" } } },
        "conv-26-q4": { phases: { evaluate: { label: "incorrect" } } },
        "conv-26-q5": { phases: { evaluate: { label: "incorrect" } } },
      },
    })
  )

  // --- per-question result JSONs: real shape pinned from
  // data/runs/aggregate-trigger-conv26-q22-fresh-20260604-001/results/conv-26-q22.json
  // (questionId/question/questionType/groundTruth/containerTag/timestamp/
  // durationMs/results — search-phase dump, no judge verdict, no evidence).
  // For bonfires-cxn, results[].metadata.utterance_hash is the CxnSearchResult
  // shape from retrieval.ts's assembleResults().
  function writeResult(id: string, questionType: string, hashes: string[]): void {
    writeFileSync(
      join(resultsDir, `${id}.json`),
      JSON.stringify({
        questionId: id,
        question: "irrelevant for classifyRun",
        questionType,
        groundTruth: "irrelevant",
        containerTag: `${id}-run`,
        timestamp: "2026-07-07T00:00:00.000Z",
        durationMs: 1,
        results: hashes.map((h) => ({
          text: "utterance",
          kind: "cxn_utterance",
          score: 1,
          metadata: { utterance_hash: h, firing_uuids: [], construct_ids: [], session: "s1" },
        })),
      })
    )
  }
  writeResult("conv-26-q1", "multi-hop", []) // correct — retrievedHashes irrelevant
  writeResult("conv-26-q2", "multi-hop", [H1]) // answered_wrong: H1 hydratable + retrieved
  writeResult("conv-26-q3", "single-hop", []) // not_retrieved: H2 hydratable, not retrieved
  writeResult("conv-26-q4", "single-hop", []) // not_hydratable: stmt-3 linked, zero plan records
  writeResult("conv-26-q5", "temporal", []) // unlinked: D9:9 has no batches message

  // --- conv-26 batches JSON: session-batched messages, metadata.dia_id +
  // timestamp ("+00:00" form) + username. D9:9 deliberately absent to
  // exercise the unlinked case.
  writeFileSync(
    join(root, "batches.json"),
    JSON.stringify([
      [
        { username: "Caroline", timestamp: "2023-05-08T13:56:00+00:00", metadata: { dia_id: "D1:1" } },
        { username: "Caroline", timestamp: "2023-05-08T13:57:00+00:00", metadata: { dia_id: "D1:2" } },
        { username: "Caroline", timestamp: "2023-05-08T13:58:00+00:00", metadata: { dia_id: "D1:3" } },
      ],
    ])
  )

  // --- activation log JSONL: ts in "Z" form (batches use "+00:00" — this is
  // the normalization the brief calls out).
  writeFileSync(
    join(root, "activation_log.jsonl"),
    [
      { statement_id: "stmt-1", speaker: "Caroline", ts: "2023-05-08T13:56:00Z" },
      { statement_id: "stmt-2", speaker: "Caroline", ts: "2023-05-08T13:57:00Z" },
      { statement_id: "stmt-3", speaker: "Caroline", ts: "2023-05-08T13:58:00Z" },
    ]
      .map((r) => JSON.stringify(r))
      .join("\n") + "\n"
  )

  // --- fold plan JSONL: stmt-1 as a bare "card" correlation_id, stmt-2 as a
  // suffixed "residual" correlation_id (both conventions covered). stmt-3 has
  // NO plan record at all — the statement still links, but yields zero
  // hashes (not_hydratable).
  writeFileSync(
    join(root, "fold_plan.jsonl"),
    [
      {
        kind: "card",
        correlation_id: "stmt-1",
        utterance: "Caroline greeted Mel.",
        event_ts: "2023-05-08T13:56:00Z",
        actor_id: "Caroline",
      },
      {
        kind: "residual",
        correlation_id: "stmt-2#r0",
        utterance: "Melanie felt supported.",
        event_ts: "2023-05-08T13:57:00Z",
        actor_id: "Caroline",
      },
    ]
      .map((r) => JSON.stringify(r))
      .join("\n") + "\n"
  )

  // --- sidecar utterance map: H1 and H2 both hydratable. ---
  writeFileSync(
    join(root, "map.json"),
    JSON.stringify({
      [H1]: { utterance: "Caroline greeted Mel.", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline", session: "s1" },
      [H2]: { utterance: "Melanie felt supported.", ts: "2023-05-08T13:57:00Z", actor_id: "Caroline", session: "s1" },
    })
  )

  return {
    runDir,
    batchesPath: join(root, "batches.json"),
    logPath: join(root, "activation_log.jsonl"),
    planPath: join(root, "fold_plan.jsonl"),
    mapPath: join(root, "map.json"),
    evidenceByQuestionId: {
      "conv-26-q1": ["D1:1"],
      "conv-26-q2": ["D1:1"],
      "conv-26-q3": ["D1:2"],
      "conv-26-q4": ["D1:3"],
      "conv-26-q5": ["D9:9"],
    },
    // "conv-26-q999" is deliberately NOT one of this run's questions (no
    // checkpoint entry, no results file) — it pins M4: loadGapInputs must
    // scope allQuestionTexts down to only the ids this run actually touched,
    // not the full LoCoMo corpus passed in here.
    questionTextById: {
      "conv-26-q1": "When did Caroline go to the LGBTQ support group?",
      "conv-26-q2": "Is that true?",
      "conv-26-q3": "Who did Melanie support?",
      "conv-26-q4": "Is it true that Caroline moved?",
      "conv-26-q5": "Did Caroline think that was true?",
      "conv-26-q999": BOGUS_QUESTION_TEXT,
    },
  }
}

// --- I2 fixture: builds a run-dir exercising both incomplete-run modes in
// isolation from the main 5-question fixture above (so the existing
// total===5 assertions aren't perturbed).
//   - conv-26-p1: complete ("correct"), has a results file — normal.
//   - conv-26-p2: checkpoint has NO evaluate label (status "pending") but
//     DOES have a results file — I2a "unevaluated".
//   - conv-26-p3: checkpoint has a completed evaluate label but NO
//     results/<id>.json on disk at all — I2b "missingResults".
function buildIncompleteFixture(): Fixture {
  const root = mkdtempSync(join(tmpdir(), "cxn-gapmap-incomplete-"))
  const runDir = join(root, "run")
  const resultsDir = join(runDir, "results")
  mkdirSync(resultsDir, { recursive: true })

  writeFileSync(
    join(runDir, "checkpoint.json"),
    JSON.stringify({
      questions: {
        "conv-26-p1": { phases: { evaluate: { label: "correct" } } },
        "conv-26-p2": { phases: { evaluate: { status: "pending" } } },
        "conv-26-p3": { phases: { evaluate: { label: "incorrect" } } },
      },
    })
  )

  function writeResult(id: string): void {
    writeFileSync(
      join(resultsDir, `${id}.json`),
      JSON.stringify({
        questionId: id,
        question: "irrelevant for classifyRun",
        questionType: "misc",
        groundTruth: "irrelevant",
        containerTag: `${id}-run`,
        timestamp: "2026-07-07T00:00:00.000Z",
        durationMs: 1,
        results: [],
      })
    )
  }
  writeResult("conv-26-p1")
  writeResult("conv-26-p2")
  // conv-26-p3 deliberately has no results/<id>.json.

  writeFileSync(join(root, "batches.json"), JSON.stringify([]))
  writeFileSync(join(root, "activation_log.jsonl"), "")
  writeFileSync(join(root, "fold_plan.jsonl"), "")
  writeFileSync(join(root, "map.json"), JSON.stringify({}))

  return {
    runDir,
    batchesPath: join(root, "batches.json"),
    logPath: join(root, "activation_log.jsonl"),
    planPath: join(root, "fold_plan.jsonl"),
    mapPath: join(root, "map.json"),
    evidenceByQuestionId: {},
    questionTextById: {
      "conv-26-p1": "p1 is complete and correct",
      "conv-26-p2": "p2 has no evaluate label yet",
      "conv-26-p3": "p3 has a label but no results file",
    },
  }
}

describe("classifyRun (via loadGapInputs, synthetic run-dir fixture)", () => {
  test("classifies one correct question and one miss of each class", async () => {
    const fixture = buildFixture()
    const inputs = await loadGapInputs(fixture)
    const report: GapReport = classifyRun(inputs)

    expect(report.overall).toEqual({ total: 5, correct: 1, score: 0.2 })
    expect(report.incomplete).toEqual({ unevaluated: 0, missingResults: 0 })
    expect(report.incompleteIds).toEqual({ unevaluated: [], missingResults: [] })
    expect(report.emptyEvidenceUnlinkedCount).toBe(0)

    expect(report.byCategory["multi-hop"]).toEqual({
      total: 2,
      correct: 1,
      missByClass: { not_hydratable: 0, not_retrieved: 0, answered_wrong: 1, unlinked: 0 },
    })
    expect(report.byCategory["single-hop"]).toEqual({
      total: 2,
      correct: 0,
      missByClass: { not_hydratable: 1, not_retrieved: 1, answered_wrong: 0, unlinked: 0 },
    })
    expect(report.byCategory["temporal"]).toEqual({
      total: 1,
      correct: 0,
      missByClass: { not_hydratable: 0, not_retrieved: 0, answered_wrong: 0, unlinked: 1 },
    })

    // byCategory keys sorted
    expect(Object.keys(report.byCategory)).toEqual(["multi-hop", "single-hop", "temporal"])

    expect(report.unlinkedEvidence).toEqual(["D9:9"])
  })

  test("evidence dia_id with no matching batches message is unlinked (chain broken before statement resolution)", async () => {
    const fixture = buildFixture()
    const inputs = await loadGapInputs(fixture)
    const report = classifyRun(inputs)
    expect(report.byCategory["temporal"]!.missByClass.unlinked).toBe(1)
  })

  test("statement resolved via ts/speaker but zero plan records is not_hydratable, not unlinked", async () => {
    const fixture = buildFixture()
    const inputs = await loadGapInputs(fixture)
    const report = classifyRun(inputs)
    // conv-26-q4 evidence D1:3 -> stmt-3 resolves via the log (ts+speaker
    // match survives the +00:00/Z normalization) but fold_plan.jsonl has no
    // record for stmt-3 at all — this must land in not_hydratable, proving
    // "linked" and "hydratable" are tracked as separate concerns.
    expect(report.byCategory["single-hop"]!.missByClass.not_hydratable).toBe(1)
  })

  test("empty evidence array classifies as unlinked with no dia_ids reported", async () => {
    const fixture = buildFixture()
    const inputs = await loadGapInputs(fixture)
    inputs.questions.push({
      questionId: "conv-26-q99",
      questionType: "world-knowledge",
      correct: false,
      evidence: [],
      retrievedHashes: [],
    })
    const report = classifyRun(inputs)
    expect(report.byCategory["world-knowledge"]!.missByClass.unlinked).toBe(1)
    expect(report.unlinkedEvidence).not.toContain(undefined as unknown as string)
    expect(report.unlinkedEvidence).toEqual(["D9:9"])
    // conv-26-q99 (pushed here with evidence: []) is the only empty-evidence
    // unlinked miss — conv-26-q5's D9:9 evidence is non-empty, it just fails
    // to link (M3).
    expect(report.emptyEvidenceUnlinkedCount).toBe(1)
  })

  test("junk-seed corpus is scoped to this run's questionIds, not the full LoCoMo corpus (M4)", async () => {
    const fixture = buildFixture()
    const inputs = await loadGapInputs(fixture)
    expect([...inputs.allQuestionTexts].sort()).toEqual(
      [
        "When did Caroline go to the LGBTQ support group?",
        "Is that true?",
        "Who did Melanie support?",
        "Is it true that Caroline moved?",
        "Did Caroline think that was true?",
      ].sort()
    )
    expect(inputs.allQuestionTexts).not.toContain(BOGUS_QUESTION_TEXT)
  })
})

describe("incomplete-run handling (I2)", () => {
  test("a checkpoint question with no evaluate label is excluded from classification and counted as incomplete.unevaluated", async () => {
    const fixture = buildIncompleteFixture()
    const inputs = await loadGapInputs(fixture)

    // conv-26-p2 must not appear as a GapQuestionResult at all.
    expect(inputs.questions.map((q) => q.questionId)).toEqual(["conv-26-p1"])
    expect(inputs.incompleteQuestions).toContainEqual({ questionId: "conv-26-p2", reason: "unevaluated" })

    const report = classifyRun(inputs)
    expect(report.incomplete.unevaluated).toBe(1)
    expect(report.incompleteIds.unevaluated).toEqual(["conv-26-p2"])
  })

  test("a checkpoint question with a completed label but no results file is counted as incomplete.missingResults, not dropped", async () => {
    const fixture = buildIncompleteFixture()
    const inputs = await loadGapInputs(fixture)

    expect(inputs.incompleteQuestions).toContainEqual({ questionId: "conv-26-p3", reason: "missingResults" })

    const report = classifyRun(inputs)
    expect(report.incomplete.missingResults).toBe(1)
    expect(report.incompleteIds.missingResults).toEqual(["conv-26-p3"])
  })

  test("overall.total counts only the questions that entered scoring (correct + classified misses), excluding both incomplete modes", async () => {
    const fixture = buildIncompleteFixture()
    const inputs = await loadGapInputs(fixture)
    const report = classifyRun(inputs)

    // Only conv-26-p1 (correct) enters scoring; p2 (unevaluated) and p3
    // (missingResults) are excluded from total/byCategory entirely.
    expect(report.overall).toEqual({ total: 1, correct: 1, score: 1 })
    expect(report.incomplete).toEqual({ unevaluated: 1, missingResults: 1 })
    expect(report.byCategory["misc"]).toEqual({
      total: 1,
      correct: 1,
      missByClass: { not_hydratable: 0, not_retrieved: 0, answered_wrong: 0, unlinked: 0 },
    })

    // The run-scoped junk corpus still includes all three ids (search ran
    // for p1/p2; p3 is tracked via checkpoint even without a results file).
    expect(inputs.allQuestionTexts.length).toBe(3)
  })
})

describe("junk-seed report", () => {
  test("literal spec (via extractTerms) is degenerate for stopword-named candidates", () => {
    const texts = [
      "When did Caroline go to the LGBTQ support group?",
      "Is that true?",
      "Who did Melanie support?",
      "Is it true that Caroline moved?",
    ]
    const report = computeJunkSeedReport(texts)
    expect(report.length).toBeLessThanOrEqual(10)
    const byName = new Map(report.map((r) => [r.name, r.questionCount]))
    // "who" and "that" are themselves in STOPWORDS, so extractTerms strips
    // them before the exact-token check ever runs — always 0. (They may not
    // even surface in the top 10, since every stopword ties at 0 and
    // alphabetically-earlier ones win the tie-break — that tie itself proves
    // the degeneracy: no stopword candidate can ever beat 0.)
    expect(byName.get("who") ?? 0).toBe(0)
    expect(byName.get("that") ?? 0).toBe(0)
    // "true" is NOT a stopword, so it survives extractTerms and gets a real,
    // non-degenerate count (2 of the 4 questions contain "true") — it must
    // win the top spot outright since every stopword is stuck at 0.
    expect(report[0]).toEqual({ name: "true", questionCount: 2 })
  })

  test("raw-token remainder shows what stopword names WOULD match without the STOPWORDS shield", () => {
    const texts = ["Is that true?", "Who did Melanie support?", "Who is that?"]
    const raw = computeRawTokenJunkCounts(texts)
    const byName = new Map(raw.map((r) => [r.name, r.questionCount]))
    expect(byName.get("who")).toBe(2)
    expect(byName.get("that")).toBe(2)
    expect(byName.get("true")).toBe(1)
  })
})
