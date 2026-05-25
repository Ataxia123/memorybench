import { describe, it, expect, mock, spyOn } from "bun:test"
import { armSearch, flattenFacts } from "./search.js"

describe("flattenFacts", () => {
  it("maps kg_delve edges to {text, score}", () => {
    const out = flattenFacts({
      edges: [
        { fact: "Alice killed Bob", score: 0.9 },
        { fact: "Bob owned a watch", score: 0.7 },
      ],
    })
    expect(out).toEqual([
      { text: "Alice killed Bob", score: 0.9, kind: "fact" },
      { text: "Bob owned a watch", score: 0.7, kind: "fact" },
    ])
  })

  it("returns [] when edges missing", () => {
    expect(flattenFacts({})).toEqual([])
  })
})

describe("armSearch", () => {
  const baseCfg = { bonfireId: "bf", arm: "vector" as const, apiUrl: "", apiKey: "" }

  it("vector arm calls vectorSearch with k=10", async () => {
    const client = {
      vectorSearch: mock(async () => [{ text: "chunk1", score: 0.8 }]),
      kgDelve: mock(async () => ({ edges: [] })),
    }
    const out = await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "vector" },
    })
    expect(client.vectorSearch.mock.calls[0][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      limit: 10,
    })
    expect(out).toEqual([{ text: "chunk1", score: 0.8, kind: "chunk" }])
  })

  it("graph arm calls kgDelve with smart=false", async () => {
    const client = {
      vectorSearch: mock(async () => []),
      kgDelve: mock(async () => ({ edges: [{ fact: "f1", score: 0.5 }] })),
    }
    await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "graph" },
    })
    expect(client.kgDelve.mock.calls[0][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 10,
      smart: false,
    })
  })

  it("smart arm calls kgDelve with smart=true node/edge recipes", async () => {
    const client = {
      vectorSearch: mock(async () => []),
      kgDelve: mock(async () => ({ edges: [{ fact: "f1", score: 0.5 }] })),
    }
    await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "smart" },
    })
    expect(client.kgDelve.mock.calls[0][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 20,
      smart: true,
      searchRecipe: "NODE_HYBRID_SEARCH_RRF",
      bfsScopes: undefined,
      rerankScopes: undefined,
    })
    expect(client.kgDelve.mock.calls[1][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 20,
      smart: true,
      searchRecipe: "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
      bfsScopes: undefined,
      rerankScopes: undefined,
    })
  })

  it("returns [] on search error rather than throwing", async () => {
    const errSpy = spyOn(console, "error").mockImplementation(() => {})
    const client = {
      vectorSearch: mock(async () => {
        throw new Error("boom")
      }),
      kgDelve: mock(async () => ({ edges: [] })),
    }
    const out = await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "vector" },
    })
    expect(out).toEqual([])
    errSpy.mockRestore()
  })

  it("hypermem arm defaults to score-ranked context and does not expose fact bookkeeping timestamps", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    const prevFactTopK = process.env.BONFIRES_HYPERMEM_FACT_TOP_K
    try {
      delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      delete process.env.BONFIRES_HYPERMEM_FACT_TOP_K
      const client = {
        hypermemSearch: mock(async () => ({
          topics: [{ score: 0.3, data: { title: "Topic", summary: "Topic summary" } }],
          episodes: [{ score: 0.2, data: { summary: "Episode summary", timestamp: "2023-05-08" } }],
          facts: [
            {
              score: 0.1,
              data: {
                content: "Grammar fact",
                timestamp: "2026-05-13T23:00:00Z",
              },
            },
            {
              score: 0.09,
              data: {
                content: "Statement fact",
                temporal: "2023-05-07",
                timestamp: "2026-05-13T23:00:00Z",
              },
            },
            {
              score: 0.08,
              data: {
                content:
                  "Caroline went to an LGBTQ support group yesterday. [2023-05-08T14:00:00.000Z] [resolved relative time: yesterday = 7 May 2023; anchor: 8 May 2023]",
                temporal: "2023-05-08; resolved relative time: yesterday = 7 May 2023",
              },
            },
            {
              score: 0.07,
              data: {
                content: "Caroline attended an LGBTQ support group on May 7, 2023.",
                temporal: "2023-05-08T14:04:00.000Z",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "when",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(client.hypermemSearch.mock.calls[0][0].outputType).toBe("111")
      expect(client.hypermemSearch.mock.calls[0][0].factTopK).toBe(14)
      expect(out.map((h) => h.kind)).toEqual([
        "community",
        "episode",
        "fact",
        "fact",
        "fact",
        "fact",
      ])
      expect(out[0].text).toBe("[TOPIC] Topic: Topic summary")
      expect(out[1].text).toBe("[EPISODE] Episode summary [occurred 8 May 2023]")
      expect(out[2].text).toBe("[FACT] Grammar fact")
      expect(out[3].text).toBe("[FACT] Statement fact [occurred 7 May 2023]")
      expect(out[4].text).toBe(
        "[FACT] Caroline went to an LGBTQ support group yesterday. [resolved relative time: yesterday = 7 May 2023; anchor: 8 May 2023]"
      )
      expect(out[5].text).toBe("[FACT] Caroline attended an LGBTQ support group on May 7, 2023.")
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
      if (prevFactTopK === undefined) delete process.env.BONFIRES_HYPERMEM_FACT_TOP_K
      else process.env.BONFIRES_HYPERMEM_FACT_TOP_K = prevFactTopK
    }
  })

  it("hypermem arm preserves explicit topic-hidden output type for token tuning runs", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          topics: [{ score: 0.4, data: { title: "Topic", summary: "Topic summary" } }],
          episodes: [{ score: 0.3, data: { summary: "Episode summary" } }],
          facts: [{ score: 0.2, data: { content: "Fact content" } }],
        })),
      }

      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "what happened",
        config: { ...baseCfg, arm: "hypermem" },
      })

      expect(client.hypermemSearch.mock.calls[0][0].outputType).toBe("011")
      expect(out.map((h) => h.kind)).toEqual(["episode", "fact"])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm can interleave direct facts with graph evidence", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "interleave"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          episodes: [{ score: 0.2, data: { summary: "Episode summary" } }],
          facts: [
            { score: 0.3, data: { content: "Direct fact one" } },
            { score: 0.2, data: { content: "Direct fact two" } },
          ],
          evidence: [
            { score: 0.9, data: { content: "Graph evidence one" } },
            { score: 0.8, data: { content: "Graph evidence two" } },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "graph evidence",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((h) => h.text)).toEqual([
        "[FACT] Direct fact one",
        "[FACT] Graph evidence one",
        "[FACT] Direct fact two",
        "[FACT] Graph evidence two",
        "[EPISODE] Episode summary",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm consumes Delve formatted context and preserves compact answer payload", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    const prevDiagnostics = process.env.BONFIRES_HYPERMEM_DIAGNOSTICS
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      process.env.BONFIRES_HYPERMEM_DIAGNOSTICS = "1"
      const client = {
        hypermemSearch: mock(async () => ({
          context: [
            "Episodes memories:",
            "## Relevant Facts:",
            "[Fact 1] Direct fact one",
            "  Time: 2023-07-15T13:53:00.000Z",
            "",
            "[Fact 2] Context-only observation fact",
            "  Time: 2023-07-15T13:57:00.000Z",
            "## Relevant Graph Facts:",
            "[Graph Fact 1] Context-only graph fact",
            "  Time: 2023-07-16T10:00:00.000Z",
          ].join("\n"),
          bonfire_id: "analytics-only",
          profile: "nlp_single_graph_v1",
          query: "observation",
          facts: [
            {
              score: 0.3,
              data: { content: "Direct fact one", temporal: "2023-07-15T13:53:00.000Z" },
            },
          ],
          evidence: [
            {
              score: 0.2,
              data: { content: "Returned graph fact", temporal: "2023-07-16T10:00:00.000Z" },
            },
          ],
          answer_context_envelope: {
            answer_candidates: [{ family: "temporal", answer_evidence: { answer_kind: "date" } }],
          },
          diagnostics: { context_token_count: 42 },
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "observation",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out).toHaveLength(1)
      expect(out[0].kind).toBe("delve_payload")
      expect(out[0].text).toContain("Context-only observation fact")
      expect(out[0].text).toContain("Context-only graph fact")
      expect(out[0].metadata?.delve_payload).toMatchObject({
        context: expect.stringContaining("Context-only observation fact"),
        answer_context_envelope: {
          answer_candidates: [{ family: "temporal" }],
        },
      })
      const payload = out[0].metadata?.delve_payload as Record<string, unknown>
      expect(payload.bonfire_id).toBeUndefined()
      expect(payload.profile).toBeUndefined()
      expect(payload.query).toBeUndefined()
      expect(payload.facts).toBeUndefined()
      expect(payload.evidence).toBeUndefined()
      expect(payload.diagnostics).toBeUndefined()
      expect(out[0].metadata?.hypermem_analytics).toEqual({
        payload_counts: { topics: 0, episodes: 0, facts: 1, evidence: 1 },
        diagnostics: { context_token_count: 42 },
      })
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
      if (prevDiagnostics === undefined) delete process.env.BONFIRES_HYPERMEM_DIAGNOSTICS
      else process.env.BONFIRES_HYPERMEM_DIAGNOSTICS = prevDiagnostics
    }
  })

  it("hypermem arm preserves server-resolved relative-time facts", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const content =
        "Melanie and her family celebrated her daughter's birthday with a concert last night."
      const client = {
        hypermemSearch: mock(async () => ({
          context: [
            "## Relevant Facts:",
            `[Fact 1] ${content} [resolved relative time: last night = 13 August 2023; anchor: 14 August 2023]`,
            "  Time: resolved relative time: last night = 13 August 2023; anchor: 14 August 2023",
          ].join("\n"),
          facts: [
            {
              score: 0.8,
              source: "statement",
              data: {
                content: `${content} [resolved relative time: last night = 13 August 2023; anchor: 14 August 2023]`,
                temporal:
                  "resolved relative time: last night = 13 August 2023; anchor: 14 August 2023",
                source_type: "statement",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "When is Melanie's daughter's birthday?",
        config: { ...baseCfg, arm: "hypermem" },
      })

      const birthdayHits = out.map((h) => h.text).filter((text) => text.includes("birthday"))
      expect(birthdayHits.length).toBe(1)
      expect(birthdayHits.every((text) => text.includes("last night = 13 August 2023"))).toBe(true)
      expect(birthdayHits.some((text) => text.includes("[occurred 14 August 2023]"))).toBe(false)
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves compact construction grammar diagnostics", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    const prevDiagnostics = process.env.BONFIRES_HYPERMEM_DIAGNOSTICS
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "001"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      process.env.BONFIRES_HYPERMEM_DIAGNOSTICS = "1"
      const client = {
        hypermemSearch: mock(async () => ({
          diagnostics: {
            construction_grammar: {
              matched_recipes: [{ recipe_kind: "TEMPORAL_ROLE_BOUND_EVENT" }],
              top_evidence_ids: ["fact-1"],
            },
          },
          facts: [
            {
              score: 0.8,
              source: "global_temporal_action_date",
              data: {
                source_type: "temporal_answer",
                event_count: 2,
                event_subject: "Alice",
                event_action: "planned",
                event_object: "coffee",
                content: "Alice planned coffee on Friday.",
                metadata: {
                  grammar_query_plan: [
                    {
                      recipe_kind: "TEMPORAL_ROLE_BOUND_EVENT",
                      matched: true,
                      trace_id: "trace-1",
                      bound_slots: { temporal_cues: ["when"], ignored: [{ nested: true }] },
                    },
                  ],
                  consumed_artifacts: [
                    {
                      kind: "temporal_edge",
                      artifact_family: "TemporalRoleBoundEvent",
                      signature: "temporal|friday",
                      trace_id: "artifact-trace-1",
                      statement_ids: ["stmt-1"],
                      source_fact_ids: ["fact-1"],
                      episode_ids: ["episode-1"],
                      source_message_ids: ["message-1"],
                      source_episode_ids: ["source-episode-1"],
                    },
                  ],
                  typed_artifacts: [
                    {
                      kind: "construction_recipe",
                      artifact_family: "AggregateSet",
                      signature: "aggregate_set|alice|planned|coffee|2",
                      trace_id: "count-trace-1",
                      aggregate_kind: "event_count",
                      support_count: 2,
                      event_count: 2,
                      event_subject: "Alice",
                      event_action: "planned",
                      event_object: "coffee",
                      source_fact_ids: ["fact-1", "fact-2"],
                    },
                    {
                      kind: "event_projection",
                      artifact_family: "EventProjection",
                      signature: "projection|alice|planned",
                    },
                  ],
                },
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "When did Alice plan coffee?",
        config: { ...baseCfg, arm: "hypermem" },
      })

      const hypermem = out[0].metadata?.hypermem as Record<string, unknown>
      const nested = hypermem.metadata as Record<string, unknown>
      expect(hypermem.event_count).toBe(2)
      expect(hypermem.event_subject).toBe("Alice")
      expect(hypermem.event_action).toBe("planned")
      expect(hypermem.event_object).toBe("coffee")
      expect(out[0].metadata?.hypermem_diagnostics).toEqual({
        construction_grammar: {
          matched_recipes: [{ recipe_kind: "TEMPORAL_ROLE_BOUND_EVENT" }],
          top_evidence_ids: ["fact-1"],
        },
      })
      expect(nested.grammar_query_plan).toEqual([
        {
          recipe_kind: "TEMPORAL_ROLE_BOUND_EVENT",
          matched: true,
          trace_id: "trace-1",
          bound_slots: { temporal_cues: ["when"], ignored: [] },
        },
      ])
      expect(nested.consumed_artifacts).toEqual([
        {
          kind: "temporal_edge",
          artifact_family: "TemporalRoleBoundEvent",
          signature: "temporal|friday",
          trace_id: "artifact-trace-1",
          statement_ids: ["stmt-1"],
          source_fact_ids: ["fact-1"],
          episode_ids: ["episode-1"],
          source_message_ids: ["message-1"],
          source_episode_ids: ["source-episode-1"],
        },
      ])
      expect(nested.typed_artifacts).toEqual([
        {
          kind: "construction_recipe",
          artifact_family: "AggregateSet",
          signature: "aggregate_set|alice|planned|coffee|2",
          trace_id: "count-trace-1",
          aggregate_kind: "event_count",
          support_count: 2,
          event_count: 2,
          event_subject: "Alice",
          event_action: "planned",
          event_object: "coffee",
          source_fact_ids: ["fact-1", "fact-2"],
        },
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
      if (prevDiagnostics === undefined) delete process.env.BONFIRES_HYPERMEM_DIAGNOSTICS
      else process.env.BONFIRES_HYPERMEM_DIAGNOSTICS = prevDiagnostics
    }
  })

  it("hypermem arm does not split resolved relative-time brackets inside linked aggregates", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.8,
              source: "entity_linkage",
              data: {
                source_type: "entity_linkage",
                temporal: "2023-07-06T13:00:00.000Z",
                content:
                  "Within Family and Relationships, Friends and Caroline are semantically linked: Caroline and her friends had a picnic last week. [resolved relative time: last week = the week before 6 July 2023; anchor: 6 July 2023]; Friends and family make a significant difference in Caroline's transition.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "When did Caroline meet friends?",
        config: { ...baseCfg, arm: "hypermem" },
      })

      expect(out[0].text).toContain("anchor: 6 July 2023]")
      expect(out[0].text).not.toContain("anchor: 6 July 2023; Friends")
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm surfaces server-side episode detail facts", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.7,
              source: "episode_detail",
              data: {
                source_type: "episode_detail",
                content: "Melanie said that Matt Patterson is very talented.",
                temporal: "2023-08-14T14:56:00.000Z",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "Who performed at the concert at Melanie's daughter's birthday?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((h) => h.text)).toEqual([
        "[FACT] Melanie said that Matt Patterson is very talented. [occurred 14 August 2023]",
      ])
      expect(out[0].metadata?.source).toBe("episode_detail")
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves server-side compact episode detail bundles", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          evidence: [
            {
              score: 0.5,
              data: {
                source_type: "graph_evidence",
                content: "Melanie enjoys hiking in the mountains.",
                temporal: "2023-07-15T14:57:00.000Z",
              },
            },
          ],
          facts: [
            {
              score: 0.92,
              source: "episode_detail",
              data: {
                source_type: "episode_detail",
                episode_detail_kind: "bundle",
                temporal: "2023-07-20T21:42:00.000Z",
                content: [
                  "Melanie's family roasts marshmallows during their camping trip.",
                  "Melanie's family tells stories around the campfire during their camping trip.",
                  "Melanie's family enjoys each other's company during their camping trip.",
                ].join("; "),
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What does Melanie do with her family on hikes?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out[0].text).toContain("roasts marshmallows")
      expect(out[0].text).toContain("tells stories around the campfire")
      expect((out[0].metadata?.hypermem as Record<string, unknown>).episode_detail_kind).toBe(
        "bundle"
      )
      expect(out[1].text).toBe(
        "[FACT] Melanie enjoys hiking in the mountains. [occurred 15 July 2023]"
      )
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm does not promote weak actor-only bundles for object questions", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "score"
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          evidence: [
            {
              score: 0.9,
              data: {
                source_type: "graph_evidence",
                content: "Melanie and her kids made a cup with a dog face on it using clay.",
              },
            },
          ],
          episodes: [
            {
              score: 0.8,
              data: {
                summary: "Melanie's family had a celebration.",
                timestamp: "2023-08-14T14:56:00.000Z",
                episode_description: [
                  "Melanie feels lucky to have her family.",
                  "Melanie saw her kids smile during the celebration.",
                  "Caroline expressed admiration for the love someone has for their kids.",
                ].join("\n"),
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What kind of pot did Mel and her kids make with clay?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out[0].text).toBe(
        "[FACT] Melanie and her kids made a cup with a dog face on it using clay."
      )
      expect(out.some((hit) => hit.kind === "fact" && hit.text.includes("celebration"))).toBe(false)
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves server-rendered entity linkage candidate items", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Events and Activities, candidate linked items: family camping trip, pottery workshop. Evidence: Melanie took her kids to a pottery workshop.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does Melanie partake in?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out[0].text).toBe(
        "[FACT] Within Events and Activities, candidate linked items: family camping trip, pottery workshop. Evidence: Melanie took her kids to a pottery workshop."
      )
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves server-focused linked evidence clauses", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "graph_evidence",
                content:
                  "Within Emotional States, Melanie and Caroline are semantically linked: Melanie was scared after the roadtrip accident.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "Would Melanie go on another roadtrip soon?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out[0].text).toBe(
        "[FACT] Within Emotional States, Melanie and Caroline are semantically linked: Melanie was scared after the roadtrip accident."
      )
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm consumes server-filtered linked evidence", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.2,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Recreational Activities, candidate linked items: camping trip. Evidence: Recreational Activities: Melanie and her family went camping.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does Melanie partake in?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Within Recreational Activities, candidate linked items: camping trip. Evidence: Recreational Activities: Melanie and her family went camping.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves server-linked evidence for compatible short names", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Symbolic Objects, candidate linked items: clay sculptures. Evidence: Melanie shared a photo of a cup with a dog face on it.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What kind of pot did Mel and her kids make with clay?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Within Symbolic Objects, candidate linked items: clay sculptures. Evidence: Melanie shared a photo of a cup with a dog face on it.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm consumes server-filtered candidate evidence", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.2,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Recreational Activities, candidate linked items: pottery workshop. Evidence: Recreational Activities: Melanie found the pottery workshop relaxing.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does Melanie partake in?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Within Recreational Activities, candidate linked items: pottery workshop. Evidence: Recreational Activities: Melanie found the pottery workshop relaxing.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm consumes server-filtered supplemental evidence", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          evidence: [{ score: 0.7, data: { content: "Melanie described the roadtrip as scary." } }],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "Would Melanie go on another roadtrip soon?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Melanie described the roadtrip as scary.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves full action evidence for what-did verb questions", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "statement",
                content:
                  "Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out[0].text).toBe(
        "[FACT] Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process."
      )
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm consumes server-filtered action-target facts", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "statement",
                content:
                  "Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm consumes server-filtered linked-action clauses", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.2,
              data: {
                source_type: "statement",
                content:
                  "Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })

  it("hypermem arm preserves score order for action facts", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011"
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      process.env.BONFIRES_HYPERMEM_RERANKER = "0"
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "statement",
                content: "Caroline plans to go do some research.",
              },
            },
            {
              score: 0.2,
              data: {
                source_type: "statement",
                content:
                  "Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Caroline plans to go do some research.",
        "[FACT] Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
      ])
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker
    }
  })
})
