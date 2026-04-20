import { describe, it, expect } from "bun:test";
import { BonfiresProvider } from "./index.js";
import type { UnifiedSession } from "../../types/unified.js";

const runLive = process.env.BONFIRES_SMOKE === "1";
const describeLive = runLive ? describe : describe.skip;

describeLive("BonfiresProvider (live)", () => {
  it("round-trips ingest → awaitIndexing → search → clear on a tiny fixture", async () => {
    const p = new BonfiresProvider();
    // Flat config per orchestrator convention (not nested under .bonfires).
    await p.initialize({
      apiUrl: process.env.BONFIRES_API_URL ?? "http://localhost:8000",
      apiKey: process.env.BONFIRES_API_KEY,
      arm: (process.env.BONFIRES_ARM ?? "vector") as "vector" | "graph" | "smart",
      // Must be a valid 24-char hex ObjectId. Each arm gets its own id to
      // avoid cross-contamination. If BONFIRES_BONFIRE_ID is set it must
      // already be a 24-char hex (ensureBonfire will use it directly).
      bonfireId: process.env.BONFIRES_BONFIRE_ID ?? "69f000000000000000smoke1",
    } as unknown as Parameters<typeof p.initialize>[0]);

    const sessions: UnifiedSession[] = [
      {
        sessionId: "smoke-s1",
        messages: [
          { role: "user", content: "I bought a red watch in Paris last summer." },
          { role: "assistant", content: "Was it expensive?" },
          { role: "user", content: "Yes, around 500 euros." },
        ],
      } as unknown as UnifiedSession,
    ];

    const ingestResult = await p.ingest(
      sessions,
      {} as unknown as Parameters<typeof p.ingest>[1],
    );
    expect(ingestResult.documentIds).toHaveLength(1);
    expect(ingestResult.documentIds[0]).toBe("smoke-s1");

    // Full pipeline: stackProcess → taxonomy → communities → ontology → grammar
    await p.awaitIndexing(
      ingestResult,
      "smoke-test",
      undefined,
    );

    const hits = await p.search(
      "what did the user buy?",
      {} as unknown as Parameters<typeof p.search>[1],
    );
    expect(Array.isArray(hits)).toBe(true);
    // Shape check only — content depends on arm + indexing success.
    // If hits is non-empty, assert the element shape.
    if (hits.length > 0) {
      expect(hits[0]).toMatchObject({ text: expect.any(String) });
    }

    await p.clear("smoke-test");
  }, 1_800_000); // 30-minute timeout for indexing
});
