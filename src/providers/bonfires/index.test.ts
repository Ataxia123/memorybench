import { describe, expect, test } from "bun:test"
import { BonfiresProvider, EXTRACTIVE_PROMPTS, buildLenientLocomoJudgePrompt } from "./index.js"

describe("Bonfires extractive prompt", () => {
  test("includes ranked list, list, exact phrase, and relative-date rules", () => {
    const prompt =
      typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
        ? EXTRACTIVE_PROMPTS.answerPrompt(
            "When did Melanie go?",
            [
              {
                text: "Resolved date: Friday before 2023-07-15 = 2023-07-14",
                score: 1,
                kind: "answer_hint",
              },
              { text: "Melanie went last Friday.", score: 0.9, kind: "chunk" },
            ],
            "2023-07-15"
          )
        : ""

    expect(prompt).toContain("ranked by relevance")
    expect(prompt).toContain("distinct supported candidates")
    expect(prompt).toContain("resolved relative time")
    expect(prompt).toContain("copy")
    expect(prompt).toContain("exact phrase")
    expect(prompt).toContain("modal or likelihood questions")
    expect(prompt).toContain("strongest ranked behavioral")
  })

  test("renders Delve HyperMem payloads into the answer-visible context", () => {
    const prompt =
      typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
        ? EXTRACTIVE_PROMPTS.answerPrompt(
            "How many events?",
            [
              {
                text: "## Relevant Facts:\n[Fact 1] Count evidence",
                score: null,
                kind: "delve_payload",
                metadata: {
                  delve_payload: {
                    context: "## Relevant Facts:\n[Fact 1] Count evidence",
                    answer_context_envelope: {
                      answer_candidates: [
                        {
                          family: "count",
                          answer_evidence: { scalar: { kind: "event_count", value: "3" } },
                        },
                      ],
                    },
                    facts: [{ score: 0.9, data: { content: "Count evidence" } }],
                  },
                },
              },
            ],
            "2023-07-15"
          )
        : ""

    expect(prompt).toContain("[Delve HyperMem Payload]")
    expect(prompt).toContain('"answer_context_envelope"')
    expect(prompt).toContain('"answer_candidates"')
    expect(prompt).toContain('"value": "3"')
    expect(prompt).toContain("treat it as the authoritative")
  })

  test("provides a Zep-style lenient LoCoMo judge prompt for comparable runs", () => {
    const prompt = buildLenientLocomoJudgePrompt(
      "What did Caroline research?",
      "Adoption agencies",
      "adoption agency, adoption process"
    ).default

    expect(prompt).toContain("be generous with your grading")
    expect(prompt).toContain("touches on the same topic")
    expect(prompt).toContain("same date or time period")
    expect(prompt).toContain("Adoption agencies")
    expect(prompt).toContain("adoption agency, adoption process")
  })

  test("uses the Zep-style judge prompt by default", () => {
    const previous = process.env.BONFIRES_JUDGE_PROMPT
    delete process.env.BONFIRES_JUDGE_PROMPT
    try {
      const provider = new BonfiresProvider()
      const prompt = provider.prompts?.judgePrompt?.(
        "What did Caroline research?",
        "Adoption agencies",
        "adoption agency, adoption process"
      ).default

      expect(prompt).toContain("be generous with your grading")
      expect(prompt).toContain("touches on the same topic")
    } finally {
      if (previous === undefined) {
        delete process.env.BONFIRES_JUDGE_PROMPT
      } else {
        process.env.BONFIRES_JUDGE_PROMPT = previous
      }
    }
  })
})
