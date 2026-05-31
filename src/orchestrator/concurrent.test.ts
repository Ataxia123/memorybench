import { describe, expect, test } from "bun:test"
import { ConcurrentExecutor } from "./concurrent"

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

describe("ConcurrentExecutor", () => {
  test("non-rate-limited execution reuses freed worker slots", async () => {
    const start = Date.now()
    const thirdStartedAfterMs: number[] = []

    await ConcurrentExecutor.execute([60, 5, 5], 2, "worker-pool-test", "test", async ({ item, index }) => {
      if (index === 2) thirdStartedAfterMs.push(Date.now() - start)
      await sleep(item)
      return item
    })

    expect(thirdStartedAfterMs[0]).toBeLessThan(45)
  })

  test("rate-limited execution preserves batch boundaries", async () => {
    const start = Date.now()
    const thirdStartedAfterMs: number[] = []

    await ConcurrentExecutor.executeBatched({
      items: [60, 5, 5],
      concurrency: 2,
      rateLimitMs: 1,
      runId: "batched-test",
      phaseName: "test",
      executeTask: async ({ item, index }) => {
        if (index === 2) thirdStartedAfterMs.push(Date.now() - start)
        await sleep(item)
        return item
      },
    })

    expect(thirdStartedAfterMs[0]).toBeGreaterThanOrEqual(45)
  })
})
