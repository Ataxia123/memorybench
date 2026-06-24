import { describe, it, expect, mock, spyOn } from "bun:test"
import { armSearch, flattenFacts } from "./search.js"

function mockArg<T>(value: unknown, call = 0, arg = 0): T {
  return ((value as { mock: { calls: unknown[][] } }).mock.calls[call] ?? [])[arg] as T
}

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
    expect(mockArg<Record<string, unknown>>(client.vectorSearch)).toEqual({
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
    expect(mockArg<Record<string, unknown>>(client.kgDelve)).toEqual({
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
    expect(mockArg<Record<string, unknown>>(client.kgDelve)).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 20,
      smart: true,
      searchRecipe: "NODE_HYBRID_SEARCH_RRF",
      bfsScopes: undefined,
      rerankScopes: undefined,
    })
    expect(mockArg<Record<string, unknown>>(client.kgDelve, 1)).toEqual({
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

  it("throws memory-kernel search errors instead of completing with empty results", async () => {
    const prevEndpoint = process.env.BONFIRES_SEARCH_ENDPOINT
    const errSpy = spyOn(console, "error").mockImplementation(() => {})
    try {
      process.env.BONFIRES_SEARCH_ENDPOINT = "memory-kernel"
      const client = {
        memoryKernelSearch: mock(async () => {
          throw new Error("FCG grammar load failed")
        }),
        hypermemSearch: mock(async () => ({ facts: [] })),
      }
      await expect(
        armSearch({
          client: client as unknown as Parameters<typeof armSearch>[0]["client"],
          query: "who",
          config: { ...baseCfg, arm: "hypermem" },
        })
      ).rejects.toThrow("FCG grammar load failed")
    } finally {
      if (prevEndpoint === undefined) delete process.env.BONFIRES_SEARCH_ENDPOINT
      else process.env.BONFIRES_SEARCH_ENDPOINT = prevEndpoint
      errSpy.mockRestore()
    }
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
      const request = mockArg<{ outputType: string; factTopK: number }>(client.hypermemSearch)
      expect(request.outputType).toBe("111")
      expect(request.factTopK).toBe(14)
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

  it("hypermem arm can route through the memory-kernel endpoint", async () => {
    const prevEndpoint = process.env.BONFIRES_SEARCH_ENDPOINT
    const prevProfile = process.env.BONFIRES_HYPERMEM_PROFILE
    const prevTopK = process.env.BONFIRES_MEMORY_KERNEL_TOP_K
    const prevCandidateLimit = process.env.BONFIRES_MEMORY_KERNEL_CANDIDATE_LIMIT
    const prevSurfaceLimit = process.env.BONFIRES_MEMORY_KERNEL_SURFACE_LIMIT
    const prevConstructLimit = process.env.BONFIRES_MEMORY_KERNEL_CONSTRUCT_CANDIDATE_LIMIT
    const prevUseFcg = process.env.BONFIRES_MEMORY_KERNEL_USE_FCG
    const prevCxnLibraryProfile = process.env.BONFIRES_MEMORY_KERNEL_CXN_LIBRARY_PROFILE
    const prevCxnRecipePreselectLimit =
      process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_LIMIT
    const prevCxnRecipePreselectMinScore =
      process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_MIN_SCORE
    const prevFcgTopicTopK = process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_TOP_K
    const prevFcgComprehensionAttemptLimit =
      process.env.BONFIRES_MEMORY_KERNEL_FCG_COMPREHENSION_ATTEMPT_LIMIT
    const prevFcgTopicSimilarityThreshold =
      process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_SIMILARITY_THRESHOLD
    const prevFcgGrammarCacheSize = process.env.BONFIRES_MEMORY_KERNEL_FCG_GRAMMAR_CACHE_SIZE
    const prevMissPolicy = process.env.BONFIRES_MEMORY_KERNEL_FCG_PRECISION_MISS_POLICY
    const prevMissLearning = process.env.BONFIRES_MEMORY_KERNEL_FCG_MISS_LEARNING
    const prevTopEvidenceK = process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNING_TOP_EVIDENCE_K
    const prevSupportThreshold = process.env.BONFIRES_MEMORY_KERNEL_FCG_ABSTRACT_SUPPORT_THRESHOLD
    const prevLearnedOverlay = process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY
    const prevLearnedOverlayMinScore =
      process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY_MIN_SCORE
    const prevEcsSearch = process.env.BONFIRES_MEMORY_KERNEL_ECS_SEARCH_ENABLED
    try {
      process.env.BONFIRES_SEARCH_ENDPOINT = "memory-kernel"
      process.env.BONFIRES_HYPERMEM_PROFILE = "nlp_single_graph_v1"
      process.env.BONFIRES_MEMORY_KERNEL_TOP_K = "7"
      process.env.BONFIRES_MEMORY_KERNEL_CANDIDATE_LIMIT = "31"
      process.env.BONFIRES_MEMORY_KERNEL_SURFACE_LIMIT = "5"
      process.env.BONFIRES_MEMORY_KERNEL_CONSTRUCT_CANDIDATE_LIMIT = "13"
      process.env.BONFIRES_MEMORY_KERNEL_USE_FCG = "1"
      process.env.BONFIRES_MEMORY_KERNEL_CXN_LIBRARY_PROFILE = "performances"
      process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_LIMIT = "5"
      process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_MIN_SCORE = "0.1"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_TOP_K = "4"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_COMPREHENSION_ATTEMPT_LIMIT = "2"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_SIMILARITY_THRESHOLD = "0.42"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_GRAMMAR_CACHE_SIZE = "12"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_PRECISION_MISS_POLICY = "continue"
      delete process.env.BONFIRES_MEMORY_KERNEL_FCG_MISS_LEARNING
      process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNING_TOP_EVIDENCE_K = "3"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_ABSTRACT_SUPPORT_THRESHOLD = "2"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY = "1"
      process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY_MIN_SCORE = "0.45"
      process.env.BONFIRES_MEMORY_KERNEL_ECS_SEARCH_ENABLED = "1"
      const client = {
        memoryKernelSearch: mock(async () => ({
          evidence: [
            {
              candidate_id: "construct:1",
              family: "construct_occurrence",
              score: 0.75,
              text: "Alice met Bob.",
              source: "fcg",
              source_ids: ["stmt-1"],
              metadata: { recipe: "topic_relation" },
            },
          ],
          diagnostics: {
            fcg_selected_topic_count: 2,
            answer_candidates: [
              {
                text: "Alice",
                answer_kind: "entity",
                source_candidate_id: "construct:1",
                source_rank: 1,
                statement_id: "stmt-1",
                answer_role: "person",
                confidence: 0.75,
              },
            ],
          },
        })),
        hypermemSearch: mock(async () => ({ facts: [] })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "who met Bob?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(client.hypermemSearch.mock.calls.length).toBe(0)
      expect(mockArg<Record<string, unknown>>(client.memoryKernelSearch)).toEqual({
        bonfireId: "bf",
        query: "who met Bob?",
        profile: "nlp_single_graph_v1",
        topK: 7,
        candidateLimit: 31,
        surfaceLimit: 5,
        constructCandidateLimit: 13,
        useFcg: true,
        hydrateGraph: true,
        embedQuery: true,
        surfaceFamilies: [],
        cxnLibraryProfile: "performances",
        cxnRecipePreselectLimit: 5,
        cxnRecipePreselectMinScore: 0.1,
        fcgTopicTopK: 4,
        fcgComprehensionAttemptLimit: 2,
        fcgTopicSimilarityThreshold: 0.42,
        fcgGrammarCacheSize: 12,
        fcgPrecisionMissPolicy: "continue",
        fcgMissLearningEnabled: true,
        fcgLearningTopEvidenceK: 3,
        fcgLearningAbstractSupportThreshold: 2,
        fcgLearnedOverlayEnabled: true,
        fcgLearnedOverlayMinScore: 0.45,
        ecsSearchEnabled: true,
      })
      expect(out).toEqual([
        {
          text: "[CONSTRUCT_OCCURRENCE] Alice met Bob.",
          score: 0.75,
          kind: "fact",
          metadata: {
            source: "fcg",
            memory_kernel: {
              candidate_id: "construct:1",
              family: "construct_occurrence",
              source_ids: ["stmt-1"],
              answer_candidates: [
                {
                  text: "Alice",
                  answer_kind: "entity",
                  source_candidate_id: "construct:1",
                  source_rank: 1,
                  statement_id: "stmt-1",
                  answer_role: "person",
                  confidence: 0.75,
                },
              ],
              metadata: { recipe: "topic_relation" },
            },
          },
        },
      ])
      expect(Object.getOwnPropertyDescriptor(out, "diagnostics")?.enumerable).toBe(false)
      expect((out as unknown as { diagnostics: unknown }).diagnostics).toEqual({
        memory_kernel: {
          fcg_selected_topic_count: 2,
          answer_candidates: [
            {
              text: "Alice",
              answer_kind: "entity",
              source_candidate_id: "construct:1",
              source_rank: 1,
              statement_id: "stmt-1",
              answer_role: "person",
              confidence: 0.75,
            },
          ],
        },
      })
    } finally {
      if (prevEndpoint === undefined) delete process.env.BONFIRES_SEARCH_ENDPOINT
      else process.env.BONFIRES_SEARCH_ENDPOINT = prevEndpoint
      if (prevProfile === undefined) delete process.env.BONFIRES_HYPERMEM_PROFILE
      else process.env.BONFIRES_HYPERMEM_PROFILE = prevProfile
      if (prevTopK === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_TOP_K
      else process.env.BONFIRES_MEMORY_KERNEL_TOP_K = prevTopK
      if (prevCandidateLimit === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_CANDIDATE_LIMIT
      else process.env.BONFIRES_MEMORY_KERNEL_CANDIDATE_LIMIT = prevCandidateLimit
      if (prevSurfaceLimit === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_SURFACE_LIMIT
      else process.env.BONFIRES_MEMORY_KERNEL_SURFACE_LIMIT = prevSurfaceLimit
      if (prevConstructLimit === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_CONSTRUCT_CANDIDATE_LIMIT
      else process.env.BONFIRES_MEMORY_KERNEL_CONSTRUCT_CANDIDATE_LIMIT = prevConstructLimit
      if (prevUseFcg === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_USE_FCG
      else process.env.BONFIRES_MEMORY_KERNEL_USE_FCG = prevUseFcg
      if (prevCxnLibraryProfile === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_CXN_LIBRARY_PROFILE
      else process.env.BONFIRES_MEMORY_KERNEL_CXN_LIBRARY_PROFILE = prevCxnLibraryProfile
      if (prevCxnRecipePreselectLimit === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_LIMIT
      else
        process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_LIMIT = prevCxnRecipePreselectLimit
      if (prevCxnRecipePreselectMinScore === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_MIN_SCORE
      else
        process.env.BONFIRES_MEMORY_KERNEL_CXN_RECIPE_PRESELECT_MIN_SCORE =
          prevCxnRecipePreselectMinScore
      if (prevFcgTopicTopK === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_TOP_K
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_TOP_K = prevFcgTopicTopK
      if (prevFcgComprehensionAttemptLimit === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_COMPREHENSION_ATTEMPT_LIMIT
      else
        process.env.BONFIRES_MEMORY_KERNEL_FCG_COMPREHENSION_ATTEMPT_LIMIT =
          prevFcgComprehensionAttemptLimit
      if (prevFcgTopicSimilarityThreshold === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_SIMILARITY_THRESHOLD
      else
        process.env.BONFIRES_MEMORY_KERNEL_FCG_TOPIC_SIMILARITY_THRESHOLD =
          prevFcgTopicSimilarityThreshold
      if (prevFcgGrammarCacheSize === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_GRAMMAR_CACHE_SIZE
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_GRAMMAR_CACHE_SIZE = prevFcgGrammarCacheSize
      if (prevMissPolicy === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_PRECISION_MISS_POLICY
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_PRECISION_MISS_POLICY = prevMissPolicy
      if (prevMissLearning === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_MISS_LEARNING
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_MISS_LEARNING = prevMissLearning
      if (prevTopEvidenceK === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNING_TOP_EVIDENCE_K
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNING_TOP_EVIDENCE_K = prevTopEvidenceK
      if (prevSupportThreshold === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_ABSTRACT_SUPPORT_THRESHOLD
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_ABSTRACT_SUPPORT_THRESHOLD = prevSupportThreshold
      if (prevLearnedOverlay === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY
      else process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY = prevLearnedOverlay
      if (prevLearnedOverlayMinScore === undefined)
        delete process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY_MIN_SCORE
      else
        process.env.BONFIRES_MEMORY_KERNEL_FCG_LEARNED_OVERLAY_MIN_SCORE =
          prevLearnedOverlayMinScore
      if (prevEcsSearch === undefined) delete process.env.BONFIRES_MEMORY_KERNEL_ECS_SEARCH_ENABLED
      else process.env.BONFIRES_MEMORY_KERNEL_ECS_SEARCH_ENABLED = prevEcsSearch
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

      expect(mockArg<{ outputType: string }>(client.hypermemSearch).outputType).toBe("011")
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
            route_total_latency_ms: 456.7,
            route_pre_response_latency_ms: 400.1,
            answer_context_envelope_ms: 12.3,
            final_context_format_ms: 4.5,
            query_embedding_ms: 123.4,
            query_embedder_call: {
              path: "provider",
              queue_wait_ms: 20.1,
              provider_ms: 103.3,
              total_ms: 123.4,
            },
            query_embedder_process_stats: {
              provider_concurrency: 1,
              cache_hits: 4,
            },
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
        route_total_latency_ms: 456.7,
        route_pre_response_latency_ms: 400.1,
        answer_context_envelope_ms: 12.3,
        final_context_format_ms: 4.5,
        query_embedding_ms: 123.4,
        query_embedder_call: {
          path: "provider",
          queue_wait_ms: 20.1,
          provider_ms: 103.3,
          total_ms: 123.4,
        },
        query_embedder_process_stats: {
          provider_concurrency: 1,
          cache_hits: 4,
        },
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
        query: "What activities does the speaker take part in?",
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
        query: "What activities does the speaker take part in?",
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
                  "Within Recreational Activities, candidate linked items: pottery workshop. Evidence: Recreational Activities: Melanie found the pottery workshop calming.",
              },
            },
          ],
        })),
      }
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does the speaker take part in?",
        config: { ...baseCfg, arm: "hypermem" },
      })
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Within Recreational Activities, candidate linked items: pottery workshop. Evidence: Recreational Activities: Melanie found the pottery workshop calming.",
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
