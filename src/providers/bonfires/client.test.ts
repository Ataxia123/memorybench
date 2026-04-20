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

  it("stackAdd posts messages to /agents/{id}/stack/add", async () => {
    const fetchMock = mock(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ added: 2 }),
      text: async () => "",
    }));
    const client = new BonfiresClient({
      apiUrl: "http://localhost:8000",
      fetchImpl: fetchMock as unknown as typeof fetch,
    });
    await client.stackAdd("agent-123", [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/agents/agent-123/stack/add");
    expect(JSON.parse((init as RequestInit).body as string).messages).toHaveLength(2);
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
