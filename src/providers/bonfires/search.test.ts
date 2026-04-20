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
      { text: "Alice killed Bob", score: 0.9 },
      { text: "Bob owned a watch", score: 0.7 },
    ]);
  });

  it("returns [] when edges missing", () => {
    expect(flattenFacts({})).toEqual([]);
  });
});

describe("armSearch", () => {
  const baseCfg = { bonfireId: "bf", arm: "vector" as const, apiUrl: "", apiKey: "" };

  it("vector arm calls vectorSearch with k=5", async () => {
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
      limit: 5,
    });
    expect(out).toEqual([{ text: "chunk1", score: 0.8 }]);
  });

  it("graph arm calls kgDelve with smart=false, autoResolveCenter=false", async () => {
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
      numResults: 5,
      smart: false,
      autoResolveCenter: false,
    });
  });

  it("smart arm calls kgDelve with smart=true", async () => {
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
      numResults: 5,
      smart: true,
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
});
