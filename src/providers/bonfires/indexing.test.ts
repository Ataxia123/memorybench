import { describe, it, expect, mock } from "bun:test";
import { runIndexingPipeline } from "./indexing.js";

describe("runIndexingPipeline", () => {
  it(
    "calls startSummaries → waitForJob → startTaxonomy → waitForJob → buildCommunities → buildGrammar in order",
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
          return { job_id: "tax1" };
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
        "buildCommunities",
        "buildGrammar",
      ]);
    },
  );
});
