import { describe, expect, test } from "bun:test"
import {
  assembleResults,
  buildCxnAnswerPrompt,
  extractTerms,
  rankFirings,
  type CandidateFiring,
} from "./retrieval"

describe("extractTerms", () => {
  test("drops stopwords, keeps unigrams and bigrams, dedupes, stable order", () => {
    const terms = extractTerms("When did Caroline adopt the puppy?")
    expect(terms).toContain("caroline")
    expect(terms).toContain("puppy")
    expect(terms).toContain("caroline adopt")
    expect(terms).not.toContain("the")
    expect(terms).not.toContain("when")
    expect(new Set(terms).size).toBe(terms.length)
  })
})

function candidate(overrides: Partial<CandidateFiring>): CandidateFiring {
  return {
    uuid: "u", constructId: "c.v1", utteranceHash: "h", ts: "2023-05-08T13:56:00Z",
    matchedSeeds: new Set(["s1"]), ...overrides,
  }
}

describe("rankFirings", () => {
  test("orders by matched seeds desc, then entrenchment desc, then ts asc, then uuid asc", () => {
    const entrenchment = new Map([["hi.v1", 50], ["lo.v1", 5]])
    const ranked = rankFirings(
      [
        candidate({ uuid: "d", constructId: "lo.v1", matchedSeeds: new Set(["s1"]) }),
        candidate({ uuid: "c", constructId: "hi.v1", matchedSeeds: new Set(["s1"]) }),
        candidate({ uuid: "b", constructId: "lo.v1", matchedSeeds: new Set(["s1", "s2"]) }),
        candidate({ uuid: "a", constructId: "unknown.v1", ts: "2023-01-01T00:00:00Z", matchedSeeds: new Set(["s1"]) }),
      ],
      entrenchment,
      10
    )
    // CORRECTED from the brief's original ["b","c","a","d"]: the tuple is
    // (matchedSeeds.size DESC, entrenchment DESC, ts ASC, uuid ASC). b has 2
    // matched seeds so it wins outright. Among the remaining 1-seed candidates
    // c/a/d, entrenchment decides: c=hi.v1(50) > d=lo.v1(5) > a=unknown.v1(0,
    // absent from the map). So the correct order is b, c, d, a — NOT b, c, a, d.
    expect(ranked.map((f) => f.uuid)).toEqual(["b", "c", "d", "a"])
  })

  test("truncates to topK deterministically", () => {
    const ranked = rankFirings(
      [candidate({ uuid: "a" }), candidate({ uuid: "b" }), candidate({ uuid: "c" })],
      new Map(),
      2
    )
    expect(ranked.length).toBe(2)
    expect(ranked.map((f) => f.uuid)).toEqual(["a", "b"])
  })
})

describe("assembleResults", () => {
  test("dedupes by hash, orders chronologically, formats [date speaker] text", () => {
    const utteranceMap = new Map([
      ["h2", { utterance: "Later thing.", ts: "2023-06-01T10:00:00Z", actor_id: "Mel", session: "s2" }],
      ["h1", { utterance: "Early thing.", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline", session: "s1" }],
    ])
    const results = assembleResults(
      [
        candidate({ uuid: "f2", utteranceHash: "h2", constructId: "b.v1" }),
        candidate({ uuid: "f1", utteranceHash: "h1", constructId: "a.v1" }),
        candidate({ uuid: "f3", utteranceHash: "h1", constructId: "c.v1" }), // same utterance, second firing
      ],
      utteranceMap,
      []
    )
    expect(results.length).toBe(2)
    expect(results[0]!.text).toBe("[2023-05-08 13:56 Caroline] Early thing.")
    expect(results[0]!.metadata.firing_uuids).toEqual(["f1", "f3"])
    expect(results[1]!.text).toBe("[2023-06-01 10:00 Mel] Later thing.")
  })
})

describe("buildCxnAnswerPrompt", () => {
  test("renders utterances + structure + question + date", () => {
    const context = [
      { text: "[2023-05-08 13:56 Caroline] Early thing.", kind: "cxn_utterance", score: 1,
        metadata: { utterance_hash: "h1", firing_uuids: ["f1"], construct_ids: ["a.v1"], session: "s1" } },
      { kind: "cxn_structure", lines: ["support(supporter=friends, person=Melanie) @ 2023-05-08T13:57:00Z [person.supported.v1]"] },
    ]
    const prompt = buildCxnAnswerPrompt("When did it happen?", context, "2023-07-01")
    expect(prompt).toContain("Early thing.")
    expect(prompt).toContain("support(supporter=friends")
    expect(prompt).toContain("When did it happen?")
    expect(prompt).toContain("2023-07-01")
  })
})
