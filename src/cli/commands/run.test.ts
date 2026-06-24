import { describe, expect, test } from "bun:test"
import { locomoConvGuardMessage, parseRunArgs } from "./run.js"

describe("run command argument parsing", () => {
  test("recovers question ids from noisy command output", () => {
    const parsed = parseRunArgs([
      "-p",
      "bonfires",
      "-b",
      "locomo",
      "--questions",
      "INFO selected targets: conv-26-q0,conv-26-q1,conv-26-q147",
    ])

    expect(parsed?.questionIds).toEqual(["conv-26-q0", "conv-26-q1", "conv-26-q147"])
  })

  test("rejects LOCOMO_CONV without a sampling path", () => {
    const parsed = parseRunArgs(["-p", "bonfires", "-b", "locomo"])

    expect(parsed).not.toBeNull()
    expect(locomoConvGuardMessage(parsed!, { LOCOMO_CONV: "conv-26" })).toContain(
      "only filters LoCoMo when --limit, --sample, or --questions is provided"
    )
  })

  test("allows LOCOMO_CONV with the conv-only limit recipe", () => {
    const parsed = parseRunArgs(["-p", "bonfires", "-b", "locomo", "--limit", "9999"])

    expect(parsed).not.toBeNull()
    expect(locomoConvGuardMessage(parsed!, { LOCOMO_CONV: "conv-26" })).toBeNull()
  })
})
