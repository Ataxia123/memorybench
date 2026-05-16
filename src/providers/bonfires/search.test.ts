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

  it("hypermem arm defaults to fact-first context and does not expose fact bookkeeping timestamps", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
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
            {
              score: 0.08,
              data: {
                content:
                  "Caroline went to an LGBTQ support group yesterday. [2023-05-08T14:00:00.000Z] [resolved relative time: yesterday = 7 May 2023; anchor: 8 May 2023]",
                temporal: "2023-05-08; resolved relative time: yesterday = 7 May 2023",
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
      expect(out.map((h) => h.kind)).toEqual(["fact", "fact", "fact", "episode"]);
      expect(out[0].text).toBe("[FACT] Grammar fact");
      expect(out[1].text).toBe("[FACT] Statement fact (event_time: 2023-05-07)");
      expect(out[2].text).toBe(
        "[FACT] Caroline went to an LGBTQ support group yesterday. [resolved relative time: yesterday = 7 May 2023; anchor: 8 May 2023]"
      );
      expect(out[3].text).toBe("[EPISODE] Episode summary (event_time: 2023-05-08)");
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm can interleave direct facts with graph evidence", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = "interleave";
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
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
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "graph evidence",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((h) => h.text)).toEqual([
        "[FACT] Direct fact one",
        "[EVIDENCE] Graph evidence one",
        "[FACT] Direct fact two",
        "[EVIDENCE] Graph evidence two",
        "[EPISODE] Episode summary",
      ]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm renders entity linkage leads as candidate items from the fact structure", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Events and Activities, family camping trip, pottery workshop, Melanie, and Caroline are semantically linked: Melanie took her kids to a pottery workshop.",
              },
            },
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does Melanie partake in?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out[0].text).toBe(
        "[FACT] Within Events and Activities, candidate linked items: family camping trip, pottery workshop. Evidence: Melanie took her kids to a pottery workshop."
      );
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm focuses linked evidence clauses using procedural query overlap", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "graph_evidence",
                content:
                  "Within Emotional States, Melanie and Caroline are semantically linked: Melanie expressed that she is glad they can be on this trip together.; Melanie was scared after the roadtrip accident.; Caroline was thankful everyone was okay.",
              },
            },
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "Would Melanie go on another roadtrip soon?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out[0].text).toBe(
        "[FACT] Within Emotional States, Melanie and Caroline are semantically linked: Melanie was scared after the roadtrip accident."
      );
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm drops linked evidence when only a different named actor appears in the body", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Recreational Activities, Running, Caroline, and Melanie are semantically linked: Recreational Activities: Caroline asks if the purple item is for walking or running.",
              },
            },
            {
              score: 0.2,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Recreational Activities, camping trip and Melanie are semantically linked: Recreational Activities: Melanie and her family went camping.",
              },
            },
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does Melanie partake in?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Within Recreational Activities, candidate linked items: camping trip. Evidence: Recreational Activities: Melanie and her family went camping.",
      ]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm drops rendered candidate evidence when the evidence body has a different actor", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Recreational Activities, beach experience and Melanie are semantically linked: Recreational Activities: Caroline went hiking last week.",
              },
            },
            {
              score: 0.2,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Recreational Activities, pottery workshop and Melanie are semantically linked: Recreational Activities: Melanie found the pottery workshop relaxing.",
              },
            },
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What activities does Melanie partake in?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Within Recreational Activities, candidate linked items: pottery workshop. Evidence: Recreational Activities: Melanie found the pottery workshop relaxing.",
      ]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm filters supplemental evidence without non-name query overlap", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
      const client = {
        hypermemSearch: mock(async () => ({
          evidence: [
            { score: 0.8, data: { content: "Melanie wants to do a family outing this summer." } },
            { score: 0.7, data: { content: "Melanie described the roadtrip as scary." } },
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "Would Melanie go on another roadtrip soon?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((hit) => hit.text)).toEqual(["[EVIDENCE] Melanie described the roadtrip as scary."]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm renders procedural action targets for what-did verb questions", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
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
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out[0].text).toBe(
        "[FACT] Action target for research: adoption agency. Evidence: Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process."
      );
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm filters action-target facts about a different named actor", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
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
            {
              score: 0.2,
              data: {
                source_type: "statement",
                content: "Melanie acknowledged that doing research and readying herself emotionally makes sense.",
              },
            },
          ],
        })),
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Action target for research: adoption agency. Evidence: Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
      ]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm ignores linked-action clauses when the query actor is only in the linkage lead", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
      const client = {
        hypermemSearch: mock(async () => ({
          facts: [
            {
              score: 0.3,
              data: {
                source_type: "entity_linkage",
                content:
                  "Within Creative Expression, Melanie and Caroline are semantically linked: Melanie planned to do research and ready herself emotionally.",
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
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Action target for research: adoption agency. Evidence: Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
      ]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

  it("hypermem arm promotes rendered action targets above generic action facts", async () => {
    const prevOutputType = process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
    const prevOrder = process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
    const prevReranker = process.env.BONFIRES_HYPERMEM_RERANKER;
    try {
      process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = "011";
      delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      process.env.BONFIRES_HYPERMEM_RERANKER = "0";
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
      };
      const out = await armSearch({
        client: client as unknown as Parameters<typeof armSearch>[0]["client"],
        query: "What did Caroline research?",
        config: { ...baseCfg, arm: "hypermem" },
      });
      expect(out.map((hit) => hit.text)).toEqual([
        "[FACT] Action target for research: adoption agency. Evidence: Caroline advises to do research and find an adoption agency or lawyer for help with the adoption process.",
        "[FACT] Caroline plans to go do some research.",
      ]);
    } finally {
      if (prevOutputType === undefined) delete process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE;
      else process.env.BONFIRES_HYPERMEM_OUTPUT_TYPE = prevOutputType;
      if (prevOrder === undefined) delete process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER;
      else process.env.BONFIRES_HYPERMEM_CONTEXT_ORDER = prevOrder;
      if (prevReranker === undefined) delete process.env.BONFIRES_HYPERMEM_RERANKER;
      else process.env.BONFIRES_HYPERMEM_RERANKER = prevReranker;
    }
  });

});
