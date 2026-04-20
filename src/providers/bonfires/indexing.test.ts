import { describe, it, expect, mock } from "bun:test";
import { runIndexingPipeline } from "./indexing.js";

describe("runIndexingPipeline", () => {
  it(
    "calls startSummaries → waitForJob → startTaxonomy → waitForJob → buildCommunities → buildOntology → createGrammar → seedGrammar in order",
    async () => {
      const order: string[] = [];
      const client = {
        startSummaries: mock(async () => {
          order.push("startSummaries");
          return { job_id: "sum1" };
        }),
        waitForJob: mock(async (_id: string, opts: { kind: string }) => {
          order.push(`wait:${opts.kind}`);
          return { state: "completed" as const };
        }),
        startTaxonomy: mock(async () => {
          order.push("startTaxonomy");
          return { job_id: "t1" };
        }),
        buildCommunities: mock(async () => {
          order.push("buildCommunities");
          return {};
        }),
        buildOntology: mock(async () => {
          order.push("buildOntology");
          return {};
        }),
        createGrammar: mock(async () => {
          order.push("createGrammar");
          return {};
        }),
        seedGrammar: mock(async () => {
          order.push("seedGrammar");
          return {};
        }),
      };
      await runIndexingPipeline({
        client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
        agentId: "agent-1",
        bonfireId: "bf-1",
      });
      expect(order).toEqual([
        "startSummaries",
        "wait:summaries",
        "startTaxonomy",
        "wait:taxonomy",
        "buildCommunities",
        "buildOntology",
        "createGrammar",
        "seedGrammar",
      ]);
    },
  );

  it("passes grammarName and seedQuery through to createGrammar and seedGrammar", async () => {
    const createGrammarArgs: unknown[] = [];
    const seedGrammarArgs: unknown[] = [];
    const client = {
      startSummaries: mock(async () => ({ job_id: "s1" })),
      waitForJob: mock(async () => ({ state: "completed" as const })),
      startTaxonomy: mock(async () => ({ job_id: "t1" })),
      buildCommunities: mock(async () => ({})),
      buildOntology: mock(async () => ({})),
      createGrammar: mock(async (args: unknown) => {
        createGrammarArgs.push(args);
        return {};
      }),
      seedGrammar: mock(async (args: unknown) => {
        seedGrammarArgs.push(args);
        return {};
      }),
    };
    await runIndexingPipeline({
      client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
      agentId: "a",
      bonfireId: "b",
      grammarName: "my-grammar",
      seedQuery: "characters and events",
    });
    expect((createGrammarArgs[0] as { grammar: string }).grammar).toBe("my-grammar");
    expect((seedGrammarArgs[0] as { kgQuery: string }).kgQuery).toBe("characters and events");
    expect((seedGrammarArgs[0] as { rule: string }).rule).toBe("entities");
    expect((seedGrammarArgs[0] as { numEntities: number }).numEntities).toBe(30);
  });
});
