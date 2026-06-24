import { describe, it, expect, mock } from "bun:test"
import { BonfiresClient, resolveBonfireObjectId } from "./client.js"

function mockCall<T extends unknown[]>(value: unknown, call = 0): T {
  return ((value as { mock: { calls: unknown[][] } }).mock.calls[call] ?? []) as unknown as T
}

describe("BonfiresClient", () => {
  it("resolves slug bonfire ids to deterministic ObjectIds", () => {
    expect(resolveBonfireObjectId("hypermem-recall-fast-cache-20260514-1638")).toBe(
      "d0674f072ef10866646ffea5"
    )
    expect(resolveBonfireObjectId("d0674f072ef10866646ffea5")).toBe("d0674f072ef10866646ffea5")
  })

  it("healthz hits /healthz with Bearer token", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ status: "ok" }),
      text: async () => "",
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      apiKey: "test-key",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })
    await client.healthz()
    const [url, init] = mockCall<[string, RequestInit]>(fetchMock)
    expect(url).toBe("http://localhost:8000/healthz")
    expect(init.headers).toMatchObject({
      Authorization: "Bearer test-key",
      "X-API-Key": "test-key",
    })
  })

  it("stackAdd posts whodunit-shaped messages to /agents/{id}/stack/add", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ added: 1 }),
      text: async () => "",
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })
    const msg = {
      id: "s1-0",
      text: "[user]: hi\n[assistant]: hello",
      userId: "agent-abc",
      chatId: "bf-000000000000000000000001",
      timestamp: "2026-01-01T00:00:00.000Z",
      role: "user",
    }
    await client.stackAdd("agent-123", [msg])
    const [url, init] = mockCall<[string, RequestInit]>(fetchMock)
    expect(url).toBe("http://localhost:8000/agents/agent-123/stack/add")
    const body = JSON.parse(init.body as string)
    expect(body.messages).toHaveLength(1)
    expect(body.messages[0]).toMatchObject({
      id: "s1-0",
      text: expect.any(String),
      userId: "agent-abc",
    })
  })

  it("ingestContent posts to /ingest_content", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ ok: true }),
      text: async () => "",
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })
    await client.ingestContent({ bonfireId: "bf-1", content: "hello world", title: "test-doc" })
    const [url, init] = mockCall<[string, RequestInit]>(fetchMock)
    expect(url).toBe("http://localhost:8000/ingest_content")
    const body = JSON.parse(init.body as string)
    expect(body).toMatchObject({ bonfire_id: "bf-1", content: "hello world", title: "test-doc" })
  })

  it("memoryKernelSearch posts FCG miss-learning controls", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ evidence: [] }),
      text: async () => "",
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })

    await client.memoryKernelSearch({
      bonfireId: "bf-1",
      query: "where is the passport",
      fcgPrecisionMissPolicy: "continue",
      fcgTopicTopK: 4,
      fcgComprehensionAttemptLimit: 2,
      fcgTopicSimilarityThreshold: 0.42,
      fcgGrammarCacheSize: 12,
      cxnLibraryProfile: "performances",
      cxnRecipePreselectLimit: 5,
      cxnRecipePreselectMinScore: 0.1,
      fcgMissLearningEnabled: true,
      fcgLearningTopEvidenceK: 3,
      fcgLearningAbstractSupportThreshold: 2,
      fcgLearnedOverlayEnabled: true,
      fcgLearnedOverlayMinScore: 0.45,
      ecsSearchEnabled: true,
    })

    const [url, init] = mockCall<[string, RequestInit]>(fetchMock)
    expect(url).toBe("http://localhost:8000/search/memory-kernel")
    const body = JSON.parse(init.body as string)
    expect(body).toMatchObject({
      bonfire_id: "bf-1",
      query: "where is the passport",
      fcg_precision_miss_policy: "continue",
      fcg_topic_top_k: 4,
      fcg_comprehension_attempt_limit: 2,
      fcg_topic_similarity_threshold: 0.42,
      fcg_grammar_cache_size: 12,
      cxn_library_profile: "performances",
      cxn_recipe_preselect_limit: 5,
      cxn_recipe_preselect_min_score: 0.1,
      fcg_miss_learning_enabled: true,
      fcg_learning_top_evidence_k: 3,
      fcg_learning_abstract_support_threshold: 2,
      fcg_learned_overlay_enabled: true,
      fcg_learned_overlay_min_score: 0.45,
      ecs_search_enabled: true,
    })
  })

  it("memoryKernelSearch defaults FCG miss-learning on", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ evidence: [] }),
      text: async () => "",
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })

    await client.memoryKernelSearch({
      bonfireId: "bf-1",
      query: "where is the passport",
    })

    const [, init] = mockCall<[string, RequestInit]>(fetchMock)
    const body = JSON.parse(init.body as string)
    expect(body.fcg_miss_learning_enabled).toBe(true)
    expect(body.fcg_learned_overlay_enabled).toBe(true)
    expect(body.fcg_comprehension_attempt_limit).toBe(3)
    expect(body.ecs_search_enabled).toBe(false)
  })

  it("hypermemStackIndex posts stack messages to the memory-kernel index endpoint", async () => {
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      apiKey: "test-key",
      fetchImpl: mock() as unknown as typeof fetch,
    })
    const reqWithCurl = mock(() => ({
      success: true,
      bonfire_id: "bf-1",
      profile: "nlp_single_graph_v1",
      diagnostics: {},
    }))
    ;(client as unknown as { reqWithCurl: typeof reqWithCurl }).reqWithCurl = reqWithCurl

    await client.hypermemStackIndex({
      bonfireId: "bf-1",
      profile: "nlp_single_graph_v1",
      stackPayloads: [
        {
          batch_idx: 7,
          batch_messages: [
            {
              id: "m-1",
              text: "Caroline left the passport on the shelf.",
              userId: "u-caroline",
              chatId: "conv-26-s1",
              sessionId: "conv-26-s1",
              timestamp: "2026-01-01T00:00:00.000Z",
              role: "user",
              username: "Caroline",
              metadata: { preserve_messages: true },
            },
          ],
        },
      ],
      initialCandidates: 123,
      useReranker: false,
    })

    const [method, path, body, opts] =
      mockCall<["GET" | "POST" | "PUT" | "DELETE", string, unknown, { timeoutMs?: number }]>(
        reqWithCurl
      )
    expect(method).toBe("POST")
    expect(path).toBe("/search/memory-kernel/index")
    expect(opts).toMatchObject({ timeoutMs: 7_200_000 })
    expect(body).toMatchObject({
      bonfire_id: "bf-1",
      profile: "nlp_single_graph_v1",
      message_batches: [
        [
          {
            content: "Caroline left the passport on the shelf.",
            speaker: "Caroline",
            timestamp: "2026-01-01T00:00:00.000Z",
            metadata: {
              preserve_messages: true,
              message_id: "m-1",
              user_id: "u-caroline",
              chat_id: "conv-26-s1",
              role: "user",
              session_id: "conv-26-s1",
              batch_idx: 7,
            },
          },
        ],
      ],
      config: {
        initial_candidates: 123,
        use_reranker: false,
      },
    })
  })

  it("seedGrammar builds correct query-string URL", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ seeded: 30 }),
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })
    await client.seedGrammar({
      bonfireId: "bf-1",
      grammar: "locomo",
      rule: "entities",
      kgQuery: "people places events",
      numEntities: 30,
    })
    const [url] = mockCall<[string]>(fetchMock)
    expect(url).toContain("/trimtabs/grammars/bf-1/seed")
    expect(url).toContain("grammar=locomo")
    expect(url).toContain("rule=entities")
    expect(url).toContain("kg_query=people")
    expect(url).toContain("num_entities=30")
  })

  it("findOrCreateAgent falls back to list on 409", async () => {
    let callCount = 0
    const fetchMock = mock(async () => {
      callCount++
      if (callCount === 1) {
        // POST /agents → 409
        return {
          ok: false,
          status: 409,
          json: async () => ({}),
          text: async () => "conflict",
        }
      }
      // GET /agents → list
      return {
        ok: true,
        status: 200,
        json: async () => ({ agents: [{ id: "existing-id", name: "my-agent" }] }),
        text: async () => "",
      }
    })
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })
    const result = await client.findOrCreateAgent({ bonfireId: "bf-1", name: "my-agent" })
    expect(result.id).toBe("existing-id")
    expect(fetchMock.mock.calls).toHaveLength(2)
    expect(mockCall<[string]>(fetchMock, 0)[0]).toBe("http://localhost:8000/agents")
    expect(mockCall<[string]>(fetchMock, 1)[0]).toBe("http://localhost:8000/agents")
  })

  it("throws on non-ok response", async () => {
    const fetchMock = mock(async () => ({
      ok: false,
      status: 500,
      json: async () => ({}),
      text: async () => "internal error",
    }))
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    })
    await expect(client.healthz()).rejects.toThrow("500")
  })
})
