import { describe, expect, test } from "bun:test"
import {
  applyTemporalBoost, blendScores, bm25Scores, buildAnswerPromptV2, buildBm25,
  denseScores, hydrationLines, isAskShape, lanePBoost, replyExpansion,
  temporalWindow, type StatementEntry,
} from "./retrieval2"

function stmt(overrides: Partial<StatementEntry>): StatementEntry {
  return {
    hash: "h", utterance: "u", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline",
    session: "s1", session_index: 0, construct_ids: ["residual.v1"], ...overrides,
  }
}

describe("bm25", () => {
  test("rare term outranks common term; absent term scores nothing", () => {
    const statements = [
      stmt({ hash: "a", utterance: "Oliver hid his bone in the slipper" }),
      stmt({ hash: "b", utterance: "They talked about the day" }),
      stmt({ hash: "c", utterance: "They talked about the bone of the day" }),
    ]
    const index = buildBm25(statements)
    // Fixture correction: the brief's query "where is the bone slipper" includes
    // "the", which (no stopword stripping, per the pinned formula) appears in ALL
    // THREE statements — so "b" would score nonzero on "the" alone, contradicting
    // "absent term scores nothing". Traced on paper: idf("the") = ln(1 + (3-3+0.5)/(3+0.5))
    // = ln(1.1428..) ≈ 0.1335 > 0, and "the" has tf>=1 in every statement including "b".
    // Dropping "the" from the query isolates the intended signal ("bone" common to a/c,
    // "slipper" rare to a only; "where"/"is" appear in no statement and are harmless
    // no-ops since their df is 0 and they're skipped). The formula is the spec; the
    // fixture's query is corrected here.
    const scores = bm25Scores(index, "where is bone slipper")
    expect(scores.get("a")! ).toBeGreaterThan(scores.get("c")!)
    expect(scores.has("b")).toBe(false)
  })
})

describe("blend", () => {
  test("min-max per lane, weighted union", () => {
    const dense = new Map([["a", 0.9], ["b", 0.1]])
    const sparse = new Map([["b", 5.0]])
    const blended = blendScores(dense, sparse, 0.7, 0.3)
    expect(blended.get("a")).toBeCloseTo(0.7)        // dense max -> 1 * 0.7
    expect(blended.get("b")).toBeCloseTo(0.3)        // dense min -> 0; sparse only entry -> 1 * 0.3
  })
})

describe("lane P", () => {
  test("distributes damped clamped aggregate similarity into members only", () => {
    const q = [1, 0]
    const aggregates = {
      cxn: new Map([["person.supported.v1", [1, 0]], ["person.other.v1", [-1, 0]]]),
      episode: new Map([["s1", [0, 1]]]),
    }
    const statements = new Map([
      ["a", stmt({ hash: "a", construct_ids: ["person.supported.v1"], session: "s1" })],
      ["b", stmt({ hash: "b", construct_ids: ["person.other.v1"], session: "s2" })],
    ])
    const { boosted } = lanePBoost(new Map([["a", 0.5], ["b", 0.5]]), q, aggregates, statements, 0.3, 0.3)
    expect(boosted.get("a")).toBeCloseTo(0.5 + 0.3 * 1 + 0.3 * 0)   // cxn sim 1, episode sim 0
    expect(boosted.get("b")).toBeCloseTo(0.5)                        // negative cxn sim clamped to 0; s2 has no aggregate
  })
})

describe("temporal gate", () => {
  test("month+year yields a window; bare 'when did' yields null", () => {
    expect(temporalWindow("What happened in June 2023?")).toEqual(
      { fromTs: "2023-06-01T00:00:00Z", toTs: "2023-06-30T23:59:59Z" })
    expect(temporalWindow("When did Caroline adopt the puppy?")).toBeNull()
  })

  test("boost applies inside window only", () => {
    const statements = new Map([
      ["in", stmt({ hash: "in", ts: "2023-06-10T00:00:00Z" })],
      ["out", stmt({ hash: "out", ts: "2023-07-10T00:00:00Z" })],
    ])
    const boosted = applyTemporalBoost(new Map([["in", 0.5], ["out", 0.5]]),
      statements, { fromTs: "2023-06-01T00:00:00Z", toTs: "2023-06-30T23:59:59Z" }, 0.15)
    expect(boosted.get("in")).toBeCloseTo(0.65)
    expect(boosted.get("out")).toBeCloseTo(0.5)
  })
})

describe("reply expansion", () => {
  test("ask-shaped parent pulls next statement in session at damped score", () => {
    const statements = new Map([
      ["ask", stmt({ hash: "ask", utterance: "Melanie asked Caroline about pets", session_index: 3, construct_ids: ["person.ask.v1"] })],
      ["reply", stmt({ hash: "reply", utterance: "Caroline said the pets are Luna and Oliver", session_index: 4 })],
    ])
    const { added } = replyExpansion(["ask"], new Map([["ask", 0.8]]), statements, 10, 0.8)
    expect(added).toEqual([{ hash: "reply", score: 0.8 * 0.8 }])
  })

  test("isAskShape: construct marker OR 'asked' text; plain statements are not", () => {
    expect(isAskShape(stmt({ construct_ids: ["person.ask.v1"] }))).toBe(true)
    expect(isAskShape(stmt({ utterance: "Caroline asked about the trip" }))).toBe(true)
    expect(isAskShape(stmt({ utterance: "Caroline enjoyed the trip" }))).toBe(false)
  })
})

describe("hydration", () => {
  test("window lines are chronological, deduped, caption-tagged", () => {
    const turns = new Map([["s1", [
      { ts: "2023-05-08T13:55:00Z", speaker: "Caroline", text: "before", blip_caption: null },
      { ts: "2023-05-08T13:56:00Z", speaker: "Caroline", text: "hit turn", blip_caption: null },
      { ts: "2023-05-08T13:57:00Z", speaker: "Melanie", text: "after", blip_caption: "a dog photo" },
    ]]])
    const statements = new Map([["h", stmt({ hash: "h", ts: "2023-05-08T13:56:00Z", session: "s1" })]])
    const lines = hydrationLines(["h"], statements, turns, 2)
    expect(lines[0]).toBe("[2023-05-08 13:55 Caroline] before")
    expect(lines[2]).toBe("[2023-05-08 13:57 Melanie] after (image: a dog photo)")
  })
})

describe("answer prompt v2", () => {
  test("renders evidence + context window + question", () => {
    const context = [
      { kind: "cxn_utterance", text: "[2023-05-08 13:56 Caroline] Early thing." },
      { kind: "cxn_context", lines: ["[2023-05-08 13:55 Caroline] before"] },
    ]
    const prompt = buildAnswerPromptV2("When?", context, "2023-07-01")
    expect(prompt).toContain("Early thing.")
    expect(prompt).toContain("CONTEXT WINDOW")
    expect(prompt).toContain("before")
    expect(prompt).toContain("When?")
    expect(prompt).toContain("2023-07-01")
  })
})
