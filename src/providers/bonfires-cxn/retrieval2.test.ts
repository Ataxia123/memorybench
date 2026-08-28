import { describe, expect, test } from "bun:test"
import {
  applyTemporalBoost, blendScores, bm25Scores, buildAnswerPromptV2, buildAnswerPromptV3, buildBm25,
  denseScores, hydrationLines, isAskShape, lanePBoost, mmrSelect, replyExpansion,
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

  test("modal 'may' does not open a window; month 'May' forms do", () => {
    expect(temporalWindow("What may have happened next?")).toBeNull()
    expect(temporalWindow("May I ask something?")).toBeNull()
    expect(temporalWindow("What happened in May?")).not.toBeNull()
    expect(temporalWindow("What happened on May 23, 2023?")).not.toBeNull()
  })

  test("february window respects leap years", () => {
    expect(temporalWindow("What happened in February 2024?")!.toTs).toBe("2024-02-29T23:59:59Z")
    expect(temporalWindow("What happened in February 2023?")!.toTs).toBe("2023-02-28T23:59:59Z")
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

describe("answer prompt evidence rendering (source_text)", () => {
  test("hit with metadata.source_text renders '[valid_at speaker] source_text' instead of the triple text", () => {
    const context = [
      {
        kind: "edge",
        text: "Caroline get support",
        metadata: {
          source_text: "Caroline, so glad you got the support!",
          valid_at: "2023-07-12T16:40:00Z",
          source_name: "Caroline",
        },
      },
    ]
    const prompt = buildAnswerPromptV3("Who got support?", context)
    expect(prompt).toContain("[2023-07-12 16:40 Caroline] Caroline, so glad you got the support!")
    expect(prompt).not.toContain("Caroline get support")
  })

  test("multiple hits hydrating to the SAME source_text render once (dedupe)", () => {
    const context = [
      {
        kind: "edge", text: "Caroline get support",
        metadata: { source_text: "Same underlying line.", valid_at: "2023-07-12T16:40:00Z", source_name: "Caroline" },
      },
      {
        kind: "edge", text: "Caroline express gratitude",
        metadata: { source_text: "Same underlying line.", valid_at: "2023-07-12T16:40:00Z", source_name: "Caroline" },
      },
    ]
    const prompt = buildAnswerPromptV3("What happened?", context)
    const occurrences = prompt.split("Same underlying line.").length - 1
    expect(occurrences).toBe(1)
  })

  test("hit with no metadata.source_text falls back to record.text", () => {
    const context = [
      { kind: "constructional", text: "[2023-05-08 13:56 Caroline] Early thing.", metadata: { actor_id: "Caroline" } },
    ]
    const prompt = buildAnswerPromptV3("When?", context)
    expect(prompt).toContain("[2023-05-08 13:56 Caroline] Early thing.")
  })

  test("source_text with no valid_at/speaker in metadata renders unstamped", () => {
    const context = [{ kind: "edge", text: "x", metadata: { source_text: "Bare line, no stamp fields." } }]
    const prompt = buildAnswerPromptV3("Q?", context)
    expect(prompt).toContain("Bare line, no stamp fields.")
    expect(prompt).not.toContain("[undefined")
  })

  test("metadata.speaker is used when source_name is absent", () => {
    const context = [
      { kind: "testimony", text: "x", metadata: { source_text: "Spoke via testimony lane.", valid_at: "2023-05-08T13:56:00Z", speaker: "Melanie" } },
    ]
    const prompt = buildAnswerPromptV3("Q?", context)
    expect(prompt).toContain("[2023-05-08 13:56 Melanie] Spoke via testimony lane.")
  })
})

describe("mmrSelect (leg 5)", () => {
  const vectors = new Map<string, number[]>([
    ["a", [1, 0]], ["b", [1, 0]], ["c", [0, 1]],
  ])
  const vectorOf = (id: string) => vectors.get(id)
  const scores = new Map<string, number>([["a", 1.0], ["b", 0.9], ["c", 0.5]])

  test("high lambda picks the diverse item over the redundant one", () => {
    // minmax: a=1, b=0.8, c=0. lambda=1: b => 0.8-1*cos(b,a)= -0.2 ; c => 0-1*0 = 0 -> c wins
    const picked = mmrSelect(scores, vectorOf, 2, 1.0).map(([id]) => id)
    expect(picked).toEqual(["a", "c"])
  })
  test("low lambda keeps score order", () => {
    // lambda=0.1: b => 0.8-0.1 = 0.7 ; c => 0 -> b wins
    const picked = mmrSelect(scores, vectorOf, 2, 0.1).map(([id]) => id)
    expect(picked).toEqual(["a", "b"])
  })
  test("returns ORIGINAL scores, selection order, and caps at pool size", () => {
    const out = mmrSelect(scores, vectorOf, 10, 0.3)
    expect(out.length).toBe(3)
    expect(out[0]).toEqual(["a", 1.0])
    expect(new Set(out.map(([id]) => id))).toEqual(new Set(["a", "b", "c"]))
  })
  test("first-pick tie breaks by id asc; missing vector throws naming the id", () => {
    const tied = new Map([["z", 1.0], ["y", 1.0]])
    const flat = (id: string) => (id === "z" || id === "y" ? [1, 0] : undefined)
    expect(mmrSelect(tied, flat, 1, 0.3)[0]![0]).toBe("y")
    expect(() => mmrSelect(new Map([["a", 1], ["ghost", 0.5]]), vectorOf, 2, 0.3)).toThrow(/ghost/)
  })
})
