import { describe, expect, test } from "bun:test"
import { computeQuestionSlice } from "./report"
import type { EvaluationResult } from "../../types/unified"

function evaluation(questionId: string, questionType: string, score: number): EvaluationResult {
  return {
    questionId,
    questionType,
    question: "Question?",
    score,
    label: score === 1 ? "correct" : "incorrect",
    explanation: "",
    hypothesis: "",
    groundTruth: "",
    searchResults: [],
    searchDurationMs: 0,
    answerDurationMs: 0,
    totalDurationMs: 0,
  }
}

describe("computeQuestionSlice", () => {
  test("excludes adversarial questions from ex-adversarial accuracy", () => {
    const slice = computeQuestionSlice([
      evaluation("q1", "single-hop", 1),
      evaluation("q2", "temporal", 0),
      evaluation("q3", "adversarial", 0),
    ])

    expect(slice).toEqual({
      total: 2,
      correct: 1,
      accuracy: 0.5,
    })
  })
})
