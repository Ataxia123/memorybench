import { describe, expect, test } from "bun:test"
import type { RunCheckpoint } from "../../types/checkpoint"
import {
  buildNativeActivationGateReport,
  buildNativeActivationRunSnapshot,
  parseNativeActivationGateArgs,
} from "./native-activation-gate.js"

function checkpoint(questionOverrides: Record<string, any>): RunCheckpoint {
  return {
    runId: "native-run",
    dataSourceRunId: "native-run",
    status: "completed",
    provider: "bonfires",
    benchmark: "locomo",
    judge: "gpt-4.1-mini",
    answeringModel: "gpt-4.1-mini",
    createdAt: "2026-06-09T00:00:00.000Z",
    updatedAt: "2026-06-09T00:00:00.000Z",
    questions: questionOverrides,
  } as RunCheckpoint
}

function question(id: string, overrides: Record<string, any> = {}): any {
  return {
    questionId: id,
    containerTag: id,
    question: "Where was the concert?",
    groundTruth: "At the venue.",
    questionType: "single-hop",
    phases: {
      ingest: { status: "completed", completedSessions: [] },
      indexing: { status: "completed" },
      search: {
        status: "completed",
        durationMs: 1200,
        resultCount: 1,
        results: [
          {
            kind: "statement_label",
            metadata: {
              memory_kernel: {
                family: "statement_label",
                metadata: { source_kind: "statement" },
              },
            },
          },
        ],
        diagnostics: {
          memory_kernel: {
            fcg_activation: {
              grammar_inventory_hash_mode: ":hash-lemma",
              query_supplier_hash_mode: ":hashed-categorial-network",
            },
            fcg_comprehend: {
              fcg_activation_scope: {
                status: "enforced",
                allowed_cxn_names: ["concert-cxn"],
                allowed_hash_keys: ["concert"],
                scoped_construct_count: 1,
                removed_construct_count: 3,
              },
            },
            retrieval_work_order: {
              episode_frontier: { episode_ids: ["episode-1"] },
            },
            graph_hydration: {
              hydrated_count: 1,
              out_of_frontier_graph_rows: 0,
            },
          },
        },
      },
      answer: { status: "completed" },
      evaluate: { status: "completed", score: 1 },
    },
    ...overrides,
  }
}

describe("native activation gate command helpers", () => {
  test("parses full gate defaults", () => {
    expect(parseNativeActivationGateArgs(["-r", "run-1"])).toMatchObject({
      runId: "run-1",
      stage: "full",
      maxSearchP95Ms: 6000,
      expectedTotal: 152,
      minCorrect: 106,
    })
  })

  test("passes when selected searches have scoped diagnostics and primary evidence", () => {
    const run = buildNativeActivationRunSnapshot(
      checkpoint({
        q1: question("q1"),
        q2: question("q2", {
          phases: {
            ...question("q2").phases,
            search: { ...question("q2").phases.search, durationMs: 1800 },
          },
        }),
      }),
      undefined,
      { stage: "probe" }
    )

    const report = buildNativeActivationGateReport(run, {
      stage: "probe",
      maxSearchP95Ms: 6000,
      expectedTotal: 2,
      minCorrect: 2,
    })

    expect(report.failures).toEqual([])
    expect(report.run.scopedDiagnosticsMissing).toBe(0)
    expect(report.run.graphOnlyEvidence).toBe(0)
  })

  test("fails empty search, graph-only evidence, missing scope, no frontier, latency, and floor gates", () => {
    const badGraphQuestion = question("q1", {
      phases: {
        ...question("q1").phases,
        search: {
          ...question("q1").phases.search,
          durationMs: 7001,
          results: [
            {
              kind: "graph_edge",
              metadata: {
                source: "graph",
                memory_kernel: {
                  family: "graph_edge",
                  metadata: {
                    source_kind: "graph_evidence",
                    evidence_tier: "episode_context",
                  },
                },
              },
            },
          ],
          diagnostics: {
            memory_kernel: {
              fcg_activation: {},
              retrieval_work_order: {
                episode_frontier: { episode_ids: [] },
              },
              graph_hydration: {
                hydrated_count: 1,
                out_of_frontier_graph_rows: 1,
              },
            },
          },
        },
        evaluate: { status: "completed", score: 0 },
      },
    })
    const emptyQuestion = question("q2", {
      phases: {
        ...question("q2").phases,
        search: {
          ...question("q2").phases.search,
          resultCount: 0,
          results: [],
        },
        evaluate: { status: "completed", score: 0 },
      },
    })
    const run = buildNativeActivationRunSnapshot(
      checkpoint({ q1: badGraphQuestion, q2: emptyQuestion }),
      undefined,
      { stage: "full" }
    )

    const report = buildNativeActivationGateReport(run, {
      stage: "full",
      maxSearchP95Ms: 6000,
      expectedTotal: 2,
      minCorrect: 1,
    })
    const failures = report.failures.join("\n")

    expect(failures).toContain("empty searches")
    expect(failures).toContain("graph-only evidence")
    expect(failures).toContain("missing scoped activation diagnostics")
    expect(failures).toContain("expanded graph evidence without an episode frontier")
    expect(failures).toContain("search p95 7001ms exceeds 6000ms")
    expect(failures).toContain("correct 0/2 is below required 1")
  })
})
