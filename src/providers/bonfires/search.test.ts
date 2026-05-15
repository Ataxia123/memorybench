import { describe, it, expect, mock, spyOn } from "bun:test";
import { armSearch, flattenFacts } from "./search.js";

describe("flattenFacts", () => {
  it("maps kg_delve edges to {text, score}", () => {
    const out = flattenFacts({
      edges: [
        { fact: "Alice killed Bob", score: 0.9 },
        { fact: "Bob owned a watch", score: 0.7 },
      ],
    });
    expect(out).toEqual([
      { text: "Alice killed Bob", score: 0.9, kind: "fact" },
      { text: "Bob owned a watch", score: 0.7, kind: "fact" },
    ]);
  });

  it("returns [] when edges missing", () => {
    expect(flattenFacts({})).toEqual([]);
  });
});

describe("armSearch", () => {
  const baseCfg = { bonfireId: "bf", arm: "vector" as const, apiUrl: "", apiKey: "" };

  it("vector arm calls vectorSearch with k=10", async () => {
    const client = {
      vectorSearch: mock(async () => [{ text: "chunk1", score: 0.8 }]),
      kgDelve: mock(async () => ({ edges: [] })),
    };
    const out = await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "vector" },
    });
    expect(client.vectorSearch.mock.calls[0][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      limit: 10,
    });
    expect(out).toEqual([{ text: "chunk1", score: 0.8, kind: "chunk" }]);
  });

  it("graph arm calls kgDelve with smart=false", async () => {
    const client = {
      vectorSearch: mock(async () => []),
      kgDelve: mock(async () => ({ edges: [{ fact: "f1", score: 0.5 }] })),
    };
    await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "graph" },
    });
    expect(client.kgDelve.mock.calls[0][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 10,
      smart: false,
    });
  });

  it("smart arm calls kgDelve with smart=true node/edge recipes", async () => {
    const client = {
      vectorSearch: mock(async () => []),
      kgDelve: mock(async () => ({ edges: [{ fact: "f1", score: 0.5 }] })),
    };
    await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "smart" },
    });
    expect(client.kgDelve.mock.calls[0][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 20,
      smart: true,
      searchRecipe: "NODE_HYBRID_SEARCH_RRF",
      bfsScopes: undefined,
      rerankScopes: undefined,
    });
    expect(client.kgDelve.mock.calls[1][0]).toEqual({
      bonfireId: "bf",
      query: "who",
      numResults: 20,
      smart: true,
      searchRecipe: "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
      bfsScopes: undefined,
      rerankScopes: undefined,
    });
  });

  it("returns [] on search error rather than throwing", async () => {
    const errSpy = spyOn(console, "error").mockImplementation(() => {});
    const client = {
      vectorSearch: mock(async () => {
        throw new Error("boom");
      }),
      kgDelve: mock(async () => ({ edges: [] })),
    };
    const out = await armSearch({
      client: client as unknown as Parameters<typeof armSearch>[0]["client"],
      query: "who",
      config: { ...baseCfg, arm: "vector" },
    });
    expect(out).toEqual([]);
    errSpy.mockRestore();
  });

  it("hypermem arm follows output_type order and does not expose fact bookkeeping timestamps", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
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
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "when",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(client.hypermemSearch.mock.calls[0][0].outputType).toBe("011");
      expect(out.map((h) => h.kind)).toEqual(["episode", "fact", "fact"]);
      expect(out[1].text).toBe("[FACT] Grammar fact");
      expect(out[2].text).toBe("[FACT] Statement fact (event_time: 2023-05-07)");
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });
});
