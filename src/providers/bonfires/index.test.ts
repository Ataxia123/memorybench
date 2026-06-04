import { describe, expect, test } from "bun:test"
import { BonfiresProvider, EXTRACTIVE_PROMPTS, buildLenientLocomoJudgePrompt } from "./index.js"

function memoryKernelAggregateHit(): Record<string, unknown> {
  return {
    text:
      "[INDEX_DOC] Melanie has 5 distinct playing clarinets events. " +
      "Evidence: Melanie plays the clarinet.; Melanie uses playing the clarinet as a way to relax. " +
      "What happened with Melanie? What is known about clarinets? noun:artifact noun:act",
    score: 0.9,
    kind: "fact",
    metadata: {
      source: "index_doc",
      memory_kernel: {
        family: "index_doc",
        metadata: {
          source_kind: "event_count_aggregate",
          evidence_anchor_type: "episode",
          timestamp: "2023-08-28T16:09:00.000Z",
        },
      },
    },
  }
}

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
    expect(prompt).toContain("month/year")
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
                    bonfire_id: "analytics-only",
                    profile: "nlp_single_graph_v1",
                    query: "How many events?",
                    context:
                      "## Relevant Topics:\n[Topic 1] Evidence-linked topic summary\n\n" +
                      "## Relevant Facts:\n[Fact 1] Count evidence\n\n" +
                      "## Answer Candidates:\n[Candidate 1] value: 3",
                    answer_context_envelope: {
                      answer_candidates: [
                        {
                          family: "count",
                          answer_evidence: { scalar: { kind: "event_count", value: "3" } },
                        },
                      ],
                    },
                    diagnostics: { context_token_count: 42 },
                    facts: [{ score: 0.9, data: { content: "Count evidence" } }],
                  },
                },
              },
            ],
            "2023-07-15"
          )
        : ""

    expect(prompt).toContain("[Delve HyperMem Context]")
    expect(prompt).toContain("Evidence-linked topic summary")
    expect(prompt).toContain("## Answer Candidates:")
    expect(prompt).toContain("value: 3")
    expect(prompt).not.toContain("[Delve Answer Candidates]")
    expect(prompt).not.toContain('"answer_context_envelope"')
    expect(prompt).not.toContain('"answer_candidates"')
    expect(prompt).not.toContain("analytics-only")
    expect(prompt).not.toContain('"profile"')
    expect(prompt).not.toContain('"query"')
    expect(prompt).not.toContain('"diagnostics"')
    expect(prompt).not.toContain('"facts"')
    expect(prompt).toContain("treat it as the authoritative")
  })

  test("keeps raw MemoryKernel hits by default for answer quality", () => {
    const previous = process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT
    delete process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT
    try {
      const prompt =
        typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
          ? EXTRACTIVE_PROMPTS.answerPrompt(
              "What instruments does Melanie play?",
              [memoryKernelAggregateHit()],
              "2023-08-28"
            )
          : ""

      expect(prompt).toContain("[INDEX_DOC] Melanie has 5 distinct playing clarinets events")
      expect(prompt).toContain("What happened with Melanie")
      expect(prompt).toContain("noun:artifact")
      expect(prompt).not.toContain("[MemoryKernel family=index_doc")
    } finally {
      if (previous === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT
      else process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT = previous
    }
  })

  test("compacts MemoryKernel hits before answer generation when enabled", () => {
    const previous = process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT
    process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT = "1"
    try {
      const prompt =
        typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
          ? EXTRACTIVE_PROMPTS.answerPrompt(
              "What instruments does Melanie play?",
              [memoryKernelAggregateHit()],
              "2023-08-28"
            )
          : ""

      expect(prompt).toContain("[MemoryKernel family=index_doc")
      expect(prompt).toContain("source=event_count_aggregate")
      expect(prompt).toContain("Melanie plays the clarinet")
      expect(prompt).not.toContain("What happened with Melanie")
      expect(prompt).not.toContain("noun:artifact")
    } finally {
      if (previous === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT
      else process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT = previous
    }
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
