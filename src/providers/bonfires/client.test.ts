import { describe, it, expect, mock } from "bun:test";
import { BonfiresClient, resolveBonfireObjectId } from "./client.js";

describe("BonfiresClient", () => {
  it("resolves slug bonfire ids to deterministic ObjectIds", () => {
    expect(resolveBonfireObjectId("hypermem-recall-fast-cache-20260514-1638")).toBe(
      "d0674f072ef10866646ffea5"
    );
    expect(resolveBonfireObjectId("d0674f072ef10866646ffea5")).toBe(
      "d0674f072ef10866646ffea5"
    );
  });

  it("healthz hits /healthz with Bearer token", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ status: "ok" }),
      text: async () => "",
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      apiKey: "test-key",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    await client.healthz();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/healthz");
    expect((init as RequestInit).headers).toMatchObject({
      Authorization: "Bearer test-key",
      "X-API-Key": "test-key",
    });
  });

  it("stackAdd posts whodunit-shaped messages to /agents/{id}/stack/add", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ added: 1 }),
      text: async () => "",
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    const msg = {
      id: "s1-0",
      text: "[user]: hi\n[assistant]: hello",
      userId: "agent-abc",
      chatId: "bf-000000000000000000000001",
      timestamp: "2026-01-01T00:00:00.000Z",
      role: "user",
    };
    await client.stackAdd("agent-123", [msg]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/agents/agent-123/stack/add");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0]).toMatchObject({ id: "s1-0", text: expect.any(String), userId: "agent-abc" });
  });

  it("ingestContent posts to /ingest_content", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ ok: true }),
      text: async () => "",
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    await client.ingestContent({ bonfireId: "bf-1", content: "hello world", title: "test-doc" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/ingest_content");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body).toMatchObject({ bonfire_id: "bf-1", content: "hello world", title: "test-doc" });
  });

  it("memoryKernelSearch posts FCG miss-learning controls", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ evidence: [] }),
      text: async () => "",
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });

    await client.memoryKernelSearch({
      bonfireId: "bf-1",
      query: "where is the passport",
      fcgPrecisionMissPolicy: "continue",
      fcgMissLearningEnabled: true,
      fcgLearningTopEvidenceK: 3,
      fcgLearningAbstractSupportThreshold: 2,
      fcgLearnedOverlayEnabled: true,
      fcgLearnedOverlayMinScore: 0.45,
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/search/memory-kernel");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body).toMatchObject({
      bonfire_id: "bf-1",
      query: "where is the passport",
      fcg_precision_miss_policy: "continue",
      fcg_miss_learning_enabled: true,
      fcg_learning_top_evidence_k: 3,
      fcg_learning_abstract_support_threshold: 2,
      fcg_learned_overlay_enabled: true,
      fcg_learned_overlay_min_score: 0.45,
    });
  });

  it("hypermemStackIndex posts stack messages to the memory-kernel index endpoint", async () => {
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      apiKey: "test-key",
      fetchImpl: mock() as unknown as typeof fetch,
    });
    const reqWithCurl = mock(() => ({
      success: true,
      bonfire_id: "bf-1",
      profile: "nlp_single_graph_v1",
      diagnostics: {},
    }));
    (client as unknown as { reqWithCurl: typeof reqWithCurl }).reqWithCurl = reqWithCurl;

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
    });

    const [method, path, body, opts] = reqWithCurl.mock.calls[0];
    expect(method).toBe("POST");
    expect(path).toBe("/search/memory-kernel/index");
    expect(opts).toMatchObject({ timeoutMs: 7_200_000 });
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
    });
  });

  it("seedGrammar builds correct query-string URL", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ seeded: 30 }),
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    await client.seedGrammar({
      bonfireId: "bf-1",
      grammar: "locomo",
      rule: "entities",
      kgQuery: "people places events",
      numEntities: 30,
    });
    const [url] = fetchMock.mock.calls[0];
    expect(url as string).toContain("/trimtabs/grammars/bf-1/seed");
    expect(url as string).toContain("grammar=locomo");
    expect(url as string).toContain("rule=entities");
    expect(url as string).toContain("kg_query=people");
    expect(url as string).toContain("num_entities=30");
  });

  it("findOrCreateAgent falls back to list on 409", async () => {
    let callCount = 0;
    const fetchMock = mock(async () => {
      callCount++;
      if (callCount === 1) {
        // POST /agents → 409
        return {
          ok: false,
          status: 409,
          json: async () => ({}),
          text: async () => "conflict",
        };
      }
      // GET /agents → list
      return {
        ok: true,
        status: 200,
        json: async () => ({ agents: [{ id: "existing-id", name: "my-agent" }] }),
        text: async () => "",
      };
    });
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    const result = await client.findOrCreateAgent({ bonfireId: "bf-1", name: "my-agent" });
    expect(result.id).toBe("existing-id");
    expect(fetchMock.mock.calls).toHaveLength(2);
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/agents");
    expect(fetchMock.mock.calls[1][0]).toBe("http://localhost:8000/agents");
  });

  it("throws on non-ok response", async () => {
    const fetchMock = mock(async () => ({
      ok: false,
      status: 500,
      json: async () => ({}),
      text: async () => "internal error",
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    await expect(client.healthz()).rejects.toThrow("500");
  });
});
