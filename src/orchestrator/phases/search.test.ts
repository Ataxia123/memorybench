import { describe, expect, test } from "bun:test"
import { extractSearchDiagnosticsForCheckpoint, slimResultsForCheckpoint } from "./search"

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

  test("stores memory-kernel diagnostics once instead of inside each checkpoint hit", () => {
    const result = {
      text: "Caroline went to the support group on 7 May 2023.",
      score: 1.23,
      kind: "fact",
      metadata: {
        source: "memory_kernel",
        memory_kernel: {
          candidate_id: "candidate-1",
          family: "statement_label",
          source_ids: ["statement-1"],
          answer_candidates: [
            {
              text: "7 May 2023",
              answer_kind: "date",
              source_candidate_id: "statement:statement-1",
              source_rank: 1,
              statement_id: "statement-1",
              answer_role: "time",
              confidence: 0.98,
              raw_document: "large raw answer candidate should not be copied",
            },
          ],
          metadata: {
            source_kind: "dialogue_answer",
            evidence_anchor_type: "statement_label",
            statement_id: "statement-1",
            resolved_statement_ids: ["statement-1", "question-1"],
            source_message_ids: ["message-1"],
            raw_document: "large raw text should not be copied",
          },
          diagnostics: {
            search_ms: 1234,
            candidate_count: 42,
            scored_count: 30,
            query_embedding: { enabled: true, vector_dimensions: 1024 },
            fcg_selection: {
              enabled: true,
              selected_count: 8,
              construct_definition_count: 100,
              selected_manifests: [
                {
                  item_id: "manifest-1",
                  topic_id: "topic-1",
                  taxonomy_label: "support",
                  similarity: 0.91,
                  construct_definition_ids: ["cxn-1", "cxn-2"],
                  image_payload: "large payload should not be copied",
                },
              ],
            },
            fcg_activation: {
              active_construct_definition_ids: ["many", "large", "ids"],
            },
            fcg_comprehend: {
              status: "precision_miss",
              attempt_count: 8,
              attempts: [{ error: "large trace should not be copied" }],
            },
            graph_hydration: { enabled: true, elapsed_ms: 41 },
            construction_learning: { enabled: false, status: "disabled" },
            aggregate_expansion: { read_count: 1, expanded_statement_count: 2 },
            answer_candidates: [
              {
                text: "7 May 2023",
                answer_kind: "date",
                source_candidate_id: "statement:statement-1",
                source_rank: 1,
                statement_id: "statement-1",
                answer_role: "time",
                source_family: "statement",
                source_message_id: "message-1",
                evidence_tier: "primary_statement",
                confidence: 0.98,
                answer_evidence: {
                  resolved_at: "2023-05-07T00:00:00.000Z",
                  ignored_array: [{ too_large: true }],
                },
                raw_document: "large raw answer candidate should not be copied",
              },
            ],
          },
        },
      },
    }

    const [slimmed] = slimResultsForCheckpoint([result]) as Array<Record<string, any>>
    const diagnostics = extractSearchDiagnosticsForCheckpoint([result]) as Record<string, any>

    expect(slimmed.text).toBe(result.text)
    expect(slimmed.kind).toBe("fact")
    expect(slimmed.metadata.source).toBe("memory_kernel")
    expect(slimmed.metadata.memory_kernel.candidate_id).toBe("candidate-1")
    expect(slimmed.metadata.memory_kernel.metadata.raw_document).toBeUndefined()
    expect(slimmed.metadata.memory_kernel.metadata.resolved_statement_ids).toEqual([
      "statement-1",
      "question-1",
    ])
    expect(slimmed.metadata.memory_kernel.answer_candidates).toEqual([
      {
        text: "7 May 2023",
        answer_kind: "date",
        source_candidate_id: "statement:statement-1",
        source_rank: 1,
        statement_id: "statement-1",
        answer_role: "time",
        confidence: 0.98,
      },
    ])
    expect(slimmed.metadata.memory_kernel.diagnostics).toBeUndefined()
    expect(diagnostics.memory_kernel.search_ms).toBe(1234)
    expect(diagnostics.memory_kernel.answer_candidates).toEqual([
      {
        text: "7 May 2023",
        answer_kind: "date",
        source_candidate_id: "statement:statement-1",
        source_rank: 1,
        statement_id: "statement-1",
        answer_role: "time",
        source_family: "statement",
        source_message_id: "message-1",
        evidence_tier: "primary_statement",
        confidence: 0.98,
        answer_evidence: {
          resolved_at: "2023-05-07T00:00:00.000Z",
        },
      },
    ])
    expect(diagnostics.memory_kernel.fcg_activation).toEqual({
      active_construct_definition_ids: ["many", "large", "ids"],
    })
    expect(diagnostics.memory_kernel.fcg_comprehend).toEqual({
      status: "precision_miss",
      attempt_count: 8,
      attempts: [{ error: "large trace should not be copied" }],
    })
    expect(diagnostics.memory_kernel.aggregate_expansion).toEqual({
      read_count: 1,
      expanded_statement_count: 2,
    })
    expect(
      diagnostics.memory_kernel.fcg_selection.selected_manifests[0].construct_definition_ids
    ).toEqual(["cxn-1", "cxn-2"])
    expect(diagnostics.memory_kernel.fcg_selection.selected_manifests[0].score).toBe(0.91)
  })

  test("extracts memory-kernel diagnostics from non-enumerable result-array metadata", () => {
    const results = [
      {
        text: "Melanie owns a book.",
        score: 0.9,
        kind: "fact",
        metadata: {
          source: "memory_kernel",
          memory_kernel: {
            candidate_id: "statement-1",
            family: "statement_label",
            source_ids: ["statement-1"],
            metadata: { statement_id: "statement-1" },
          },
        },
      },
    ] as unknown[]
    Object.defineProperty(results, "diagnostics", {
      value: {
        memory_kernel: {
          search_ms: 456,
          aggregate_expansion: { read_count: 1, expanded_statement_count: 1 },
        },
      },
      enumerable: false,
    })

    const [slimmed] = slimResultsForCheckpoint(results) as Array<Record<string, any>>
    const diagnostics = extractSearchDiagnosticsForCheckpoint(results) as Record<string, any>

    expect(JSON.stringify(results)).not.toContain("aggregate_expansion")
    expect(slimmed.metadata.memory_kernel.diagnostics).toBeUndefined()
    expect(diagnostics.memory_kernel.search_ms).toBe(456)
    expect(diagnostics.memory_kernel.aggregate_expansion).toEqual({
      read_count: 1,
      expanded_statement_count: 1,
    })
  })
})
