import { describe, expect, test } from "bun:test"
import {
  answerDirective, directivePreamble, combineSparse, isFallback, seededTermWeights,
  strataBoost, temporalWindowFromDates, type Comprehension,
} from "./affordance"
import { bm25ScoresWeighted, buildBm25, buildAnswerPromptV2, buildAnswerPromptV3, monthWindow, type StatementEntry } from "./retrieval2"

const statement = (hash: string, cxns: string[]): StatementEntry => ({
  hash, utterance: "u", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline",
  session: "s1", session_index: 0, construct_ids: cxns,
})
const comp = (over: Partial<Comprehension>): Comprehension => ({
  probe: "", matched_cxn_ids: [], bound_cxn_ids: [],
  operators: { neg: false, modal: null }, wh_slot: null, fillers: [], date_fillers: [], ...over,
})

describe("strataBoost", () => {
  test("boosts by matched-fraction, ignores residual, leaves others", () => {
    const scores = new Map([["a", 0.5], ["b", 0.5], ["c", 0.5]])
    const statements = new Map([
      ["a", statement("a", ["person.ask.v1", "person.paint.v1"])],
      ["b", statement("b", ["residual.v1"])],
      ["c", statement("c", ["person.paint.v1"])],
    ])
    const out = strataBoost(scores, statements, ["person.ask.v1", "person.paint.v1"], 0.3)
    expect(out.get("a")).toBeCloseTo(0.5 + 0.3, 10)          // 2/2 matched
    expect(out.get("b")).toBeCloseTo(0.5, 10)                 // residual never keys
    expect(out.get("c")).toBeCloseTo(0.5 + 0.15, 10)          // 1/2 matched
  })
  test("empty matched set is a no-op", () => {
    const scores = new Map([["a", 0.5]])
    const out = strataBoost(scores, new Map([["a", statement("a", ["x"])]]), [], 0.3)
    expect(out.get("a")).toBe(0.5)
  })
})

describe("temporalWindowFromDates", () => {
  test("month+year -> that month, leap year respected via monthWindow", () => {
    expect(temporalWindowFromDates([{ text: "May 2023", year: 2023, month: 5 }], 2023))
      .toEqual(monthWindow(2023, 4))
    expect(monthWindow(2024, 1).toTs).toBe("2024-02-29T23:59:59Z")
  })
  test("month-only uses default year; year-only spans the year; empty -> null", () => {
    expect(temporalWindowFromDates([{ text: "April", year: null, month: 4 }], 2023))
      .toEqual(monthWindow(2023, 3))
    expect(temporalWindowFromDates([{ text: "2022", year: 2022, month: null }], 2023))
      .toEqual({ fromTs: "2022-01-01T00:00:00Z", toTs: "2022-12-31T23:59:59Z" })
    expect(temporalWindowFromDates([], 2023)).toBeNull()
  })
})

describe("seeded sparse lane", () => {
  test("weights: entity > verb, max wins on collision, tokenized lemmas", () => {
    const weights = seededTermWeights(
      [{ lemma: "melanie", role: "subj", entity: true },
       { lemma: "paint", role: "verb", entity: false },
       { lemma: "melanie", role: "obj", entity: false }],
      2.0, 1.0)
    expect(weights.get("melanie")).toBe(2.0)
    expect(weights.get("paint")).toBe(1.0)
  })
  test("bm25ScoresWeighted scales term contributions", () => {
    const index = buildBm25([statement("a", []), { ...statement("b", []), utterance: "melanie paints" }])
    const weighted = bm25ScoresWeighted(index, new Map([["melanie", 2.0]]))
    const plain = bm25ScoresWeighted(index, new Map([["melanie", 1.0]]))
    expect(weighted.get("b")).toBeCloseTo((plain.get("b") ?? 0) * 2, 10)
  })
  test("combineSparse = minMax(nat) + w*minMax(seed)", () => {
    const out = combineSparse(new Map([["a", 2], ["b", 4]]), new Map([["b", 1]]), 0.5)
    expect(out.get("a")).toBeCloseTo(0, 10)
    expect(out.get("b")).toBeCloseTo(1 + 0.5 * 1, 10)
  })
})

describe("answer directive + fallback + prompt v3", () => {
  test("closed map", () => {
    expect(answerDirective("when")).toContain("date or time")
    expect(answerDirective("list")).toContain("ALL distinct items")
    expect(answerDirective("how_many")).toContain("quantity")
    expect(answerDirective(null)).toBeNull()
  })
  test("v2 strings: inference, list, when, how_long; how_many/entity unchanged", () => {
    expect(answerDirective("inference")).toBe(
      "This is an inference question. Reason from the evidence to a definite answer (e.g. yes/no or likely/unlikely) with a brief reason. Do NOT answer 'Not enough information' if the evidence supports a reasonable inference."
    )
    expect(answerDirective("list")).toBe(
      "The question asks for multiple items. Enumerate ALL distinct items supported by the evidence and the context window; do not stop at the first. Do not add items of a kind the question did not ask about."
    )
    expect(answerDirective("when")).toBe(
      "The question asks for a specific date or time. Resolve relative references in the evidence ('last Saturday', 'the week before') against that message's own timestamp, then answer with the most specific absolute date supported."
    )
    expect(answerDirective("how_long")).toBe(
      "Answer with the duration AND its absolute anchor (e.g. 'since 2016'), derived from the evidence timestamps if needed."
    )
    expect(answerDirective("how_many")).toBe("Answer with a specific quantity or duration.")
    expect(answerDirective("what")).toBe("Answer with the specific entity or fact, concisely.")
    expect(answerDirective(null)).toBeNull()
  })
  test("directivePreamble is the pinned two-sentence string", () => {
    expect(directivePreamble()).toBe(
      "State the concrete fact(s) in the evidence's own words rather than a vague paraphrase. Answer 'Not enough information' ONLY when the evidence and context contain nothing relevant to the question."
    )
  })
  test("isFallback true only when everything is empty", () => {
    expect(isFallback(comp({}))).toBe(true)
    expect(isFallback(comp({ wh_slot: "what" }))).toBe(false)
    expect(isFallback(comp({ matched_cxn_ids: ["x"] }))).toBe(false)
  })
  test("prompt v3 without directive is byte-identical to v2; with directive adds one block", () => {
    const context = [
      { kind: "cxn_utterance", text: "[2023-05-08 13:56 Caroline] hi" },
      { kind: "cxn_context", lines: ["[2023-05-08 13:55 Mel] hey"] },
    ]
    expect(buildAnswerPromptV3("Q?", context, "2023-06-01")).toBe(buildAnswerPromptV2("Q?", context, "2023-06-01"))
    const withDirective = [...context.slice(0, 1),
      { kind: "cxn_context", lines: ["l"], directive: answerDirective("when") }]
    const rendered = buildAnswerPromptV3("Q?", withDirective)
    expect(rendered).toContain("ANSWER DIRECTIVE:")
    expect(rendered).toContain("date or time")
  })
})
