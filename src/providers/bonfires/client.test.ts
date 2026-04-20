import { describe, it, expect, mock } from "bun:test";
import { BonfiresClient } from "./client.js";

describe("BonfiresClient", () => {
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
