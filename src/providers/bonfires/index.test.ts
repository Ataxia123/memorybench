import { describe, expect, test } from "bun:test"
import { EXTRACTIVE_PROMPTS } from "./index.js"

describe("Bonfires extractive prompt", () => {
  test("includes ranked list, list, exact phrase, and relative-date rules", () => {
    const prompt =
      typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
        ? EXTRACTIVE_PROMPTS.answerPrompt(
            "When did Melanie go?",
            [
              { text: "Resolved date: Friday before 2023-07-15 = 2023-07-14", score: 1, kind: "answer_hint" },
              { text: "Melanie went last Friday.", score: 0.9, kind: "chunk" },
            ],
            "2023-07-15"
          )
        : ""

    expect(prompt).toContain("ranked by relevance")
    expect(prompt).toContain("distinct candidate")
    expect(prompt).toContain("resolved relative time")
    expect(prompt).toContain("copy")
    expect(prompt).toContain("exact phrase")
  })
})
