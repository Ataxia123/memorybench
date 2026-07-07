import { describe, expect, test } from "bun:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { loadArtifacts } from "./artifacts"
import type { CxnConfig } from "./config"

function fixtureConfig(): CxnConfig {
  const dir = mkdtempSync(join(tmpdir(), "cxn-artifacts-"))
  writeFileSync(
    join(dir, "census.json"),
    JSON.stringify({
      per_card: [
        { construct_id: "person.supported.v1", bound: 44, suppressed: 0 },
        { construct_id: "person.ask.v1", bound: 72, suppressed: 0 },
      ],
    })
  )
  writeFileSync(
    join(dir, "fold_plan.jsonl"),
    '{"kind":"card"}\n{"kind":"residual"}\n{"kind":"residual"}\n'
  )
  const mapPath = join(dir, "map.json")
  writeFileSync(
    mapPath,
    JSON.stringify({
      abc123abc123abc1: { utterance: "hi", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline", session: "s1" },
    })
  )
  return {
    neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
    groupId: "g", artifactsDir: dir, utteranceMapPath: mapPath,
    expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
  }
}

describe("loadArtifacts", () => {
  test("loads map, entrenchment, and plan count", async () => {
    const artifacts = await loadArtifacts(fixtureConfig())
    expect(artifacts.planRecordCount).toBe(3)
    expect(artifacts.entrenchmentByConstruct.get("person.ask.v1")).toBe(72)
    expect(artifacts.utteranceMap.get("abc123abc123abc1")?.actor_id).toBe("Caroline")
  })
})
