import { describe, it, expect, mock } from "bun:test";
import { runIndexingPipeline } from "./indexing.js";

describe("runIndexingPipeline", () => {
  it(
    "calls startSummaries → waitForJob → startTaxonomy → waitForJob → updateLabels → buildCommunities → buildGrammar in order",
    async () => {
      const order: string[] = [];
      const client = {
        startSummaries: mock(async () => {
          order.push("startSummaries");
          return { job_id: "sum1" };
        }),
        waitForJob: mock(async (_id: string, opts: { kind: string }) => {
          order.push(`wait:${opts.kind}`);
          return opts.kind === "taxonomy"
            ? { state: "completed" as const, metadata: { result: { run_id: "run-xyz" } } }
            : { state: "completed" as const };
        }),
        startTaxonomy: mock(async () => {
          order.push("startTaxonomy");
          return { job_id: "tax1" };
        }),
        updateLabels: mock(async (_bonfireId: string, _runId: string) => {
          order.push("updateLabels");
          return {};
        }),
        buildCommunities: mock(async () => {
          order.push("buildCommunities");
          return {};
        }),
        buildGrammar: mock(async () => {
          order.push("buildGrammar");
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
        "updateLabels",
        "buildCommunities",
        "buildGrammar",
      ]);
    },
  );

  it("throws if the taxonomy job returns no run_id in its result", async () => {
    const client = {
      startSummaries: mock(async () => ({ job_id: "sum1" })),
      waitForJob: mock(async (_id: string, opts: { kind: string }) => {
        return opts.kind === "taxonomy"
          ? { state: "completed" as const, metadata: { result: {} } }
          : { state: "completed" as const };
      }),
      startTaxonomy: mock(async () => ({ job_id: "tax1" })),
      updateLabels: mock(async () => ({})),
      buildCommunities: mock(async () => ({})),
      buildGrammar: mock(async () => ({})),
    };
    await expect(
      runIndexingPipeline({
        client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
        agentId: "a",
        bonfireId: "b",
      }),
    ).rejects.toThrow(/no run_id/);
  });

  it("passes the taxonomy run_id through to updateLabels", async () => {
    const updateArgs: Array<[string, string]> = [];
    const client = {
      startSummaries: mock(async () => ({ job_id: "sum1" })),
      waitForJob: mock(async (_id: string, opts: { kind: string }) => {
        return opts.kind === "taxonomy"
          ? { state: "completed" as const, metadata: { result: { run_id: "tax-run-42" } } }
          : { state: "completed" as const };
      }),
      startTaxonomy: mock(async () => ({ job_id: "tax1" })),
      updateLabels: mock(async (bonfireId: string, runId: string) => {
        updateArgs.push([bonfireId, runId]);
        return {};
      }),
      buildCommunities: mock(async () => ({})),
      buildGrammar: mock(async () => ({})),
    };
    await runIndexingPipeline({
      client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
      agentId: "a",
      bonfireId: "my-bonfire",
    });
    expect(updateArgs).toEqual([["my-bonfire", "tax-run-42"]]);
  });
});
