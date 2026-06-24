import { describe, expect, mock, test } from "bun:test"
import { existsSync } from "fs"
import {
  BonfiresProvider,
  EXTRACTIVE_PROMPTS,
  __bonfiresProviderStateForTests,
  buildLenientLocomoJudgePrompt,
} from "./index.js"
import { resolveBonfireObjectId } from "./client.js"

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
              {
                text: "[STATEMENT] Caroline: My grandma is from Sweden.",
                score: 0.95,
                kind: "fact",
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
    expect(prompt).toContain("either/or preference questions")
    expect(prompt).toContain("shared generic category")
    expect(prompt).toContain("role and ownership slots strict")
    expect(prompt).toContain("different person")
    expect(prompt).toContain("[STATEMENT speaker=Caroline] Caroline:")
    expect(prompt).toContain("Speaker tags are ownership evidence")
    expect(prompt).toContain("subject, owner, relation, and event")
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

  test("renders MemoryKernel context packet sections before ranked fallback", () => {
    const prompt =
      typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
        ? EXTRACTIVE_PROMPTS.answerPrompt(
            "Who met Sam?",
            [
              {
                text: "[STATEMENT] Caroline met Sam in Quito.",
                score: 0.91,
                kind: "fact",
                metadata: {
                  source: "memory-kernel",
                  memory_kernel: {
                    candidate_id: "statement-1",
                    family: "statement",
                    context_packet: {
                      sections: [
                        {
                          name: "main_answer",
                          evidence: [
                            {
                              candidate_id: "statement-1",
                              role: "person",
                              rank: 1,
                              score: 0.91,
                            },
                          ],
                        },
                        {
                          name: "location_context",
                          evidence: [
                            {
                              candidate_id: "statement-2",
                              role: "location",
                              rank: 2,
                              score: 0.72,
                            },
                          ],
                        },
                      ],
                    },
                    metadata: {
                      source_kind: "statement_label",
                      statement_id: "statement-1",
                    },
                  },
                },
              },
              {
                text: "[STATEMENT] The meeting happened in Quito.",
                score: 0.72,
                kind: "fact",
                metadata: {
                  source: "memory-kernel",
                  memory_kernel: {
                    candidate_id: "statement-2",
                    family: "statement",
                    metadata: {
                      source_kind: "statement_label",
                      statement_id: "statement-2",
                    },
                  },
                },
              },
            ],
            "2023-08-23"
          )
        : ""

    expect(prompt).toContain("<MAIN_ANSWER>")
    expect(prompt).toContain("role=person")
    expect(prompt).toContain("Caroline met Sam in Quito")
    expect(prompt).toContain("<LOCATION_CONTEXT>")
    expect(prompt).toContain("role=location")
    expect(prompt).toContain("The meeting happened in Quito")
    expect(prompt).not.toContain("Context (ranked by relevance; keep this order):\n1.")
  })

  test("caps MemoryKernel context packet rendering with the ranked context budget", () => {
    const previous = process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET
    process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET = "1200"
    try {
      const longTail = " packet-detail".repeat(1200)
      const prompt =
        typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
          ? EXTRACTIVE_PROMPTS.answerPrompt(
              "Who met Sam?",
              [
                {
                  text: `[STATEMENT] Caroline met Sam in Quito.${longTail}`,
                  score: 0.91,
                  kind: "fact",
                  metadata: {
                    source: "memory-kernel",
                    memory_kernel: {
                      candidate_id: "statement-1",
                      family: "statement",
                      answer_candidates: [
                        {
                          text: "Caroline",
                          answer_kind: "entity",
                          answer_role: "person",
                          source_candidate_id: "statement-1",
                          source_rank: 1,
                          statement_id: "statement-1",
                          confidence: 0.91,
                        },
                      ],
                      context_packet: {
                        sections: [
                          {
                            name: "main_answer",
                            evidence: [
                              {
                                candidate_id: "statement-1",
                                role: "person",
                                rank: 1,
                                score: 0.91,
                              },
                            ],
                          },
                        ],
                      },
                      metadata: {
                        source_kind: "statement_label",
                        statement_id: "statement-1",
                      },
                    },
                  },
                },
              ],
              "2023-08-23"
            )
          : ""

      expect(prompt).toContain("<ANSWER_CANDIDATES>")
      expect(prompt).toContain("<MAIN_ANSWER>")
      expect(prompt).toContain("Caroline met Sam in Quito")
      expect(prompt).toContain("</MAIN_ANSWER>")
      const contextSection = prompt.split("Question Date:")[0] || prompt
      expect(contextSection.length).toBeLessThan(6500)
    } finally {
      if (previous === undefined) delete process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET
      else process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET = previous
    }
  })

  test("renders MemoryKernel answer candidates before raw evidence", () => {
    const prompt =
      typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
        ? EXTRACTIVE_PROMPTS.answerPrompt(
            "Where is the projector?",
            [
              {
                text: "[STATEMENT] The projector is in the cabinet.",
                score: 0.98,
                kind: "fact",
                metadata: {
                  source: "memory-kernel",
                  memory_kernel: {
                    candidate_id: "statement-1",
                    family: "statement",
                    answer_candidates: [
                      {
                        text: "cabinet",
                        answer_kind: "entity",
                        answer_role: "location",
                        source_candidate_id: "statement-1",
                        source_rank: 1,
                        statement_id: "stmt-projector",
                        confidence: 0.98,
                      },
                    ],
                    metadata: {
                      statement_id: "stmt-projector",
                    },
                  },
                },
              },
            ],
            "2023-08-23"
          )
        : ""

    expect(prompt).toContain("<ANSWER_CANDIDATES>")
    expect(prompt).toContain("answer_kind=entity")
    expect(prompt).toContain("answer_role=location")
    expect(prompt).toContain("statement=stmt-projector")
    expect(prompt).toContain("source=statement-1")
    expect(prompt).toContain("cabinet")
    expect(prompt).toContain("kernel-selected")
    expect(prompt).toContain("Kernel answer candidate priority")
    expect(prompt.indexOf("<ANSWER_CANDIDATES>")).toBeLessThan(
      prompt.indexOf("1. [STATEMENT] The projector is in the cabinet.")
    )
    expect(prompt.indexOf("Kernel answer candidate priority")).toBeLessThan(
      prompt.indexOf("Answer:")
    )
  })

  test("caps ranked MemoryKernel context while preserving candidates and top evidence", () => {
    const previous = process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET
    process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET = "1200"
    try {
      const longTail = " tail-detail".repeat(900)
      const prompt =
        typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
          ? EXTRACTIVE_PROMPTS.answerPrompt(
              "Where is the projector?",
              [
                {
                  text: `[STATEMENT] The projector is in the cabinet.${longTail}`,
                  score: 0.98,
                  kind: "fact",
                  metadata: {
                    source: "memory-kernel",
                    memory_kernel: {
                      candidate_id: "statement-1",
                      family: "statement",
                      answer_candidates: [
                        {
                          text: "cabinet",
                          answer_kind: "entity",
                          answer_role: "location",
                          source_candidate_id: "statement-1",
                          source_rank: 1,
                          statement_id: "stmt-projector",
                          confidence: 0.98,
                        },
                      ],
                    },
                  },
                },
                {
                  text: "[STATEMENT] Rank two evidence should survive.",
                  score: 0.9,
                  kind: "fact",
                },
                {
                  text: "[STATEMENT] Rank three evidence should survive.",
                  score: 0.8,
                  kind: "fact",
                },
                {
                  text: "[STATEMENT] Rank four evidence should survive.",
                  score: 0.7,
                  kind: "fact",
                },
                {
                  text: "[STATEMENT] Tail evidence should be omitted.",
                  score: 0.6,
                  kind: "fact",
                },
              ],
              "2023-08-23"
            )
          : ""

      expect(prompt).toContain("<ANSWER_CANDIDATES>")
      expect(prompt).toContain("cabinet")
      expect(prompt).toContain("1. [STATEMENT] The projector is in the cabinet.")
      expect(prompt).toContain("Rank two evidence should survive")
      expect(prompt).toContain("Rank four evidence should survive")
      expect(prompt).not.toContain("Tail evidence should be omitted")
      const contextSection = prompt.split("Question Date:")[0] || prompt
      expect(contextSection.length).toBeLessThan(6500)
    } finally {
      if (previous === undefined) delete process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET
      else process.env.BONFIRES_MEMORYBENCH_CONTEXT_TOKEN_BUDGET = previous
    }
  })

  test("renders MemoryKernel temporal answer scope metadata in compact context", () => {
    const previous = process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT
    process.env.BONFIRES_MEMORY_KERNEL_COMPACT_CONTEXT = "1"
    try {
      const prompt =
        typeof EXTRACTIVE_PROMPTS.answerPrompt === "function"
          ? EXTRACTIVE_PROMPTS.answerPrompt(
              "When did Caroline apply to adoption agencies?",
              [
                {
                  text: "[INDEX_DOC] Caroline applied to adoption agencies this week as the first step towards becoming a mom.",
                  score: 1.1,
                  kind: "fact",
                  metadata: {
                    source: "index_doc",
                    memory_kernel: {
                      family: "index_doc",
                      metadata: {
                        source_kind: "statement",
                        evidence_anchor_type: "statement_label",
                        timestamp: "2023-08-23T15:31:00.000Z",
                        statement_id: "statement-1",
                        answer_temporal_scope: {
                          surface: "week of 23 August 2023",
                          granularity: "week",
                          kind: "anchored_relative_period",
                        },
                      },
                    },
                  },
                },
              ],
              "2023-08-23"
            )
          : ""

      expect(prompt).toContain('date_scope="week of 23 August 2023"')
      expect(prompt).toContain("Caroline applied to adoption agencies this week")
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

  test("awaitIndexing reports selected episode ids after pipeline completion", async () => {
    const previousArm = process.env.BONFIRES_ARM
    const previousStackV2 = process.env.BONFIRES_STACK_V2
    const previousNoDoc = process.env.BONFIRES_STACK_V2_NO_DOC
    const previousDirectIndex = process.env.BONFIRES_HYPERMEM_DIRECT_INDEX
    const previousProfile = process.env.BONFIRES_HYPERMEM_PROFILE
    try {
      process.env.BONFIRES_ARM = "hypermem"
      process.env.BONFIRES_STACK_V2 = "1"
      process.env.BONFIRES_STACK_V2_NO_DOC = "1"
      process.env.BONFIRES_HYPERMEM_DIRECT_INDEX = "1"
      process.env.BONFIRES_HYPERMEM_PROFILE = "nlp_single_graph_v1"

      const provider = new BonfiresProvider()
      const providerHarness = provider as unknown as {
        client: {
          hypermemStackIndex: ReturnType<typeof mock>
        }
        config: { bonfireId: string }
        agentId: string
        indexingDone: boolean
        sessionsForKg: []
        persist: () => void
      }
      providerHarness.client = {
        hypermemStackIndex: mock(async () => ({
          success: true,
          bonfire_id: "bf-1",
          profile: "nlp_single_graph_v1",
          diagnostics: {},
        })),
      }
      providerHarness.config = { bonfireId: "bf-1" }
      providerHarness.agentId = "agent-1"
      providerHarness.indexingDone = false
      providerHarness.sessionsForKg = []
      providerHarness.persist = () => undefined
      const progress: Array<{ completedIds: string[]; failedIds: string[]; total: number }> = []

      await provider.awaitIndexing(
        { documentIds: ["session-1", "session-2"], taskIds: ["task-1"] },
        "container",
        (event) => progress.push(event)
      )

      expect(progress).toEqual([
        { completedIds: ["session-1", "session-2", "task-1"], failedIds: [], total: 3 },
      ])
      expect(providerHarness.client.hypermemStackIndex).toHaveBeenCalledTimes(1)
    } finally {
      if (previousArm === undefined) delete process.env.BONFIRES_ARM
      else process.env.BONFIRES_ARM = previousArm
      if (previousStackV2 === undefined) delete process.env.BONFIRES_STACK_V2
      else process.env.BONFIRES_STACK_V2 = previousStackV2
      if (previousNoDoc === undefined) delete process.env.BONFIRES_STACK_V2_NO_DOC
      else process.env.BONFIRES_STACK_V2_NO_DOC = previousNoDoc
      if (previousDirectIndex === undefined) delete process.env.BONFIRES_HYPERMEM_DIRECT_INDEX
      else process.env.BONFIRES_HYPERMEM_DIRECT_INDEX = previousDirectIndex
      if (previousProfile === undefined) delete process.env.BONFIRES_HYPERMEM_PROFILE
      else process.env.BONFIRES_HYPERMEM_PROFILE = previousProfile
    }
  })
})

describe("Bonfires provider state cache", () => {
  test("force clears resolved bonfire cache before loading provider state", () => {
    const bonfireId = resolveBonfireObjectId(`force-cache-test-${Date.now()}-${Math.random()}`)
    const state = __bonfiresProviderStateForTests

    state.clearState(bonfireId)
    try {
      state.saveState(bonfireId, {
        ingestedSessionIds: ["session-1"],
        indexingDone: false,
        sessionsForKg: [],
      })

      expect(existsSync(state.cachePathFor(bonfireId))).toBe(true)

      const loaded = state.hydrateState(bonfireId, true)

      expect(loaded).toEqual({
        ingestedSessionIds: [],
        indexingDone: false,
        sessionsForKg: [],
      })
      expect(existsSync(state.cachePathFor(bonfireId))).toBe(false)
    } finally {
      state.clearState(bonfireId)
    }
  })
})
