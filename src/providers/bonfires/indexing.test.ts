import { describe, it, expect, mock, spyOn } from "bun:test";
import { runIndexingPipeline } from "./indexing.js";

describe("runIndexingPipeline", () => {
  it("calls stackProcess → taxonomy → labeling → buildGrammar → buildCommunities in order", async () => {
    const order: string[] = [];
    const client = {
      stackProcess: mock(async () => {
        order.push("stackProcess");
        return { task_id: "s1" };
      }),
      waitForJob: mock(async (_id: string, opts: { kind: string }) => {
        order.push(`wait:${opts.kind}`);
        return { state: "completed" as const };
      }),
      startTaxonomy: mock(async () => {
        order.push("startTaxonomy");
        return { job_id: "t1" };
      }),
      startLabeling: mock(async () => {
        order.push("startLabeling");
        return { job_id: "l1" };
      }),
      buildGrammar: mock(async () => {
        order.push("buildGrammar");
        return { entities: 10 };
      }),
      buildCommunities: mock(async () => {
        order.push("buildCommunities");
        return {};
      }),
    };
    await runIndexingPipeline({
      client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
      agentId: "agent-1",
      bonfireId: "bf-1",
    });
    expect(order).toEqual([
      "stackProcess",
      "wait:stack_processing",
      "startTaxonomy",
      "wait:taxonomy",
      "startLabeling",
      "wait:labeling",
      "buildGrammar",
      "buildCommunities",
    ]);
  });

  it("logs warning but does not throw when buildGrammar returns 0 entities", async () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    const client = {
      stackProcess: mock(async () => ({ task_id: "s1" })),
      waitForJob: mock(async () => ({ state: "completed" as const })),
      startTaxonomy: mock(async () => ({ job_id: "t1" })),
      startLabeling: mock(async () => ({ job_id: "l1" })),
      buildGrammar: mock(async () => ({ entities: 0 })),
      buildCommunities: mock(async () => ({})),
    };
    await runIndexingPipeline({
      client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
      agentId: "a",
      bonfireId: "b",
    });
    expect(warn.mock.calls.length).toBe(1);
    expect(warn.mock.calls[0][0]).toContain("0 entities");
    warn.mockRestore();
  });
});
