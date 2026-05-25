import { describe, expect, test } from "bun:test"
import { slimResultsForCheckpoint } from "./search"

describe("slimResultsForCheckpoint", () => {
  test("keeps Delve payload answer-visible in result files but slims checkpoint copies", () => {
    const result = {
      text: "server rendered context",
      score: null,
      kind: "delve_payload",
      metadata: {
        hypermem_diagnostics: { context_token_count: 123 },
        delve_payload: {
          bonfire_id: "bf",
          profile: "nlp_single_graph_v1",
          query: "how many?",
          context: "server rendered context",
          facts: [{ data: { content: "large fact" } }],
          evidence: [{ data: { content: "large evidence" } }],
          episodes: [{ data: { summary: "large episode" } }],
          topics: [{ data: { title: "large topic" } }],
          diagnostics: { construction_grammar: { large: true } },
          answer_context_envelope: {
            selected_answer_candidates: [{ family: "count", status: "complete" }],
            answer_candidates: [{ family: "count" }],
            selected_facts: [{ content: "large selected fact" }],
            graph_hydration: { fact_count: 1 },
          },
        },
      },
    }

    const [slimmed] = slimResultsForCheckpoint([result]) as Array<Record<string, any>>

    expect(slimmed.text).toBe("server rendered context")
    expect(slimmed.kind).toBe("delve_payload")
    expect(slimmed.metadata.hypermem_diagnostics).toEqual({ context_token_count: 123 })
    expect(slimmed.metadata.delve_payload).toBeUndefined()
    expect(slimmed.metadata.delve_payload_summary.counts).toEqual({
      topics: 1,
      episodes: 1,
      facts: 1,
      evidence: 1,
    })
    expect(
      slimmed.metadata.delve_payload_summary.answer_context_envelope.selected_answer_candidates
    ).toEqual([{ family: "count", status: "complete" }])
    expect(
      slimmed.metadata.delve_payload_summary.answer_context_envelope.selected_facts
    ).toBeUndefined()
  })
})
