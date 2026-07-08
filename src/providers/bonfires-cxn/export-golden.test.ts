import { describe, expect, test } from "bun:test"
import { GOLDEN_QUESTION_IDS, buildGoldenEntry, toSparseTop40 } from "./scripts/cxn-export-golden"

describe("golden export", () => {
  test("pinned question set is exactly the 20 ids", () => {
    expect(GOLDEN_QUESTION_IDS.length).toBe(20)
    expect(GOLDEN_QUESTION_IDS[8]).toBe("conv-26-q60")
  })
  test("buildGoldenEntry shape from injected stages", () => {
    const entry = buildGoldenEntry({
      qid: "conv-26-q60",
      question: "q?",
      comprehension: {
        probe: "",
        matched_cxn_ids: [],
        bound_cxn_ids: [],
        operators: { neg: false, modal: null },
        wh_slot: "list",
        fillers: [],
        date_fillers: [],
      },
      sparseTop40: [["h1", 1.5]],
      seededSparseTop40: [["h1", 1.5]],
      denseScores: new Map([["h1", 0.5]]),
      final20: ["h1"],
      directive: null,
      affordancesFired: ["q:seed"],
      mmrFired: true,
    })
    expect(entry.final20).toEqual(["h1"])
    expect(entry.dense_scores.h1).toBeCloseTo(0.5, 12)
  })

  test("all 20 pinned ids match conv-26-q<index>", () => {
    for (const qid of GOLDEN_QUESTION_IDS) {
      expect(qid).toMatch(/^conv-26-q\d+$/)
    }
  })

  test("buildGoldenEntry threads every field through unchanged (pure passthrough)", () => {
    const comprehension = {
      probe: "melanie plays",
      matched_cxn_ids: ["person.plays.v1"],
      bound_cxn_ids: ["person.plays.v1"],
      operators: { neg: false, modal: null },
      wh_slot: "list" as const,
      fillers: [{ lemma: "melanie", role: "subj" as const, entity: true }],
      date_fillers: [],
    }
    const entry = buildGoldenEntry({
      qid: "conv-26-q60",
      question: "What instruments does Melanie play?",
      comprehension,
      sparseTop40: [
        ["b", 2],
        ["a", 1],
      ],
      seededSparseTop40: [
        ["b", 3],
        ["a", 1],
      ],
      denseScores: new Map([
        ["b", 0.2],
        ["a", 0.9],
      ]),
      final20: ["a", "b"],
      directive: "some directive",
      affordancesFired: ["q:seed", "q:answer"],
      mmrFired: false,
    })
    expect(entry.qid).toBe("conv-26-q60")
    expect(entry.question).toBe("What instruments does Melanie play?")
    expect(entry.comprehension).toEqual(comprehension)
    expect(entry.sparse_top40).toEqual([
      ["b", 2],
      ["a", 1],
    ])
    expect(entry.seeded_sparse_top40).toEqual([
      ["b", 3],
      ["a", 1],
    ])
    expect(entry.dense_scores).toEqual({ a: 0.9, b: 0.2 })
    expect(entry.final20).toEqual(["a", "b"])
    expect(entry.directive).toBe("some directive")
    expect(entry.affordances_fired).toEqual(["q:seed", "q:answer"])
    expect(entry.mmr_fired).toBe(false)
  })

  test("dense_scores keys are sorted ascending", () => {
    const entry = buildGoldenEntry({
      qid: "conv-26-q0",
      question: "q?",
      comprehension: {
        probe: "",
        matched_cxn_ids: [],
        bound_cxn_ids: [],
        operators: { neg: false, modal: null },
        wh_slot: null,
        fillers: [],
        date_fillers: [],
      },
      sparseTop40: [],
      seededSparseTop40: [],
      denseScores: new Map([
        ["zeta", 0.1],
        ["alpha", 0.2],
        ["mid", 0.3],
      ]),
      final20: [],
      directive: null,
      affordancesFired: [],
      mmrFired: false,
    })
    expect(Object.keys(entry.dense_scores)).toEqual(["alpha", "mid", "zeta"])
  })
})

describe("toSparseTop40", () => {
  test("sorts score-desc, hash-asc tiebreak, rounds to 6 decimals, caps at 40", () => {
    const scores = new Map<string, number>()
    for (let i = 0; i < 45; i++) scores.set(`h${i}`, i)
    scores.set("tieB", 10)
    scores.set("tieA", 10)
    const top = toSparseTop40(scores)
    expect(top.length).toBe(40)
    expect(top[0]).toEqual(["h44", 44])
    // both tied entries score 10; "tieA" < "tieB" lexicographically, so it sorts first among equals
    const tieIndex = top.findIndex(([hash]) => hash === "tieA" || hash === "tieB")
    expect(top[tieIndex]![0]).toBe("tieA")
  })

  test("rounds to 6 decimal places", () => {
    const top = toSparseTop40(new Map([["h", 1 / 3]]))
    expect(top[0]).toEqual(["h", 0.333333])
  })
})
