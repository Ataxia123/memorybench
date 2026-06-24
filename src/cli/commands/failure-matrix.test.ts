import { describe, expect, test } from "bun:test"
import { failureMode, parseArgs } from "./failure-matrix.js"

describe("failure matrix command helpers", () => {
  test("parses adversarial exclusion flags", () => {
    expect(parseArgs(["-r", "run-1", "--exclude-type", "adversarial,control"])).toMatchObject({
      runId: "run-1",
      excludeTypes: ["adversarial", "control"],
    })
  })

  test("buckets non-correct rows by answer hit position", () => {
    const base = {
      questionId: "q1",
      questionType: "single-hop",
      score: 0,
      resultCount: 10,
      topologyParticipated: true,
      top20FamilyCounts: {},
    }

    expect(failureMode({ ...base, hitAtK: 1, answerHitRank: 1 })).toBe("answer_top1_hit")
    expect(failureMode({ ...base, hitAtK: 1, answerHitRank: 4 })).toBe("answer_hit_not_top1")
    expect(failureMode({ ...base, hitAtK: 0 })).toBe("retrieval_no_hit")
    expect(failureMode({ ...base, resultCount: 0, hitAtK: 0 })).toBe("retrieval_no_hit")
    expect(failureMode({ ...base, score: 1, hitAtK: 0 })).toBe("clear")
  })
})
