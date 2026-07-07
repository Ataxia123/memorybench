import { describe, expect, test } from "bun:test"
import { BonfiresCxnProvider, type CxnDeps } from "./index"
import { SEED_ENTITIES, SEEDS_TO_FIRINGS, NEIGHBORS_TO_FIRINGS, FIRING_STRUCTURES } from "./cypher"
import type { CxnArtifacts } from "./artifacts"
import type { CxnConfig } from "./config"

const CFG: CxnConfig = {
  neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
  groupId: "g", artifactsDir: "/x", utteranceMapPath: "/x/m.json",
  expectedNodes: 1, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
}

const ARTIFACTS: CxnArtifacts = {
  utteranceMap: new Map([
    ["h1", { utterance: "Caroline adopted a puppy.", ts: "2023-05-08T13:56:00Z", actor_id: "Caroline", session: "s1" }],
  ]),
  entrenchmentByConstruct: new Map([["person.adopt.v1", 10]]),
  planRecordCount: 1,
}

function fakeDeps(log: string[]): CxnDeps {
  return {
    runCypher: async (query, params) => {
      log.push(query)
      if (query === SEED_ENTITIES) {
        expect(params.terms).toContain("puppy")
        return [{ uuid: "e-puppy", name: "puppy", degree: 4 }]
      }
      if (query === SEEDS_TO_FIRINGS)
        return [{
          uuid: "f1", name: "person.adopt.v1@2026-01-01T00:00:00+00:00",
          attributes: JSON.stringify({ utterance_hash: "h1", ts: "2023-05-08T13:56:00Z", predicate: "adopt", actor_id: "Caroline" }),
          matchedSeeds: ["e-puppy"],
        }]
      if (query === NEIGHBORS_TO_FIRINGS) return []
      if (query === FIRING_STRUCTURES)
        return [{ firingUuid: "f1", firingName: "person.adopt.v1@2026-01-01T00:00:00+00:00",
                  eventUuid: "ev1", eventAttributes: JSON.stringify({ predicate: "adopt" }),
                  role: "PERSON", filler: "Caroline" }]
      throw new Error(`unexpected query: ${query.slice(0, 40)}`)
    },
  }
}

describe("search", () => {
  test("walks seeds→firings, hydrates, returns utterance results + one structure item", async () => {
    const log: string[] = []
    const provider = new BonfiresCxnProvider(CFG, ARTIFACTS, fakeDeps(log))
    const results = await provider.search("When did Caroline adopt the puppy?", { containerTag: "x" })

    const utterances = results.filter((r) => (r as { kind?: string }).kind === "cxn_utterance")
    const structures = results.filter((r) => (r as { kind?: string }).kind === "cxn_structure")
    expect(utterances.length).toBe(1)
    expect((utterances[0] as { text: string }).text).toBe("[2023-05-08 13:56 Caroline] Caroline adopted a puppy.")
    expect((utterances[0] as { metadata: { construct_ids: string[] } }).metadata.construct_ids).toEqual(["person.adopt.v1"])
    expect(structures.length).toBe(1)
    expect((structures[0] as { lines: string[] }).lines[0]).toContain("adopt(PERSON=Caroline)")
    expect(log).toEqual([SEED_ENTITIES, SEEDS_TO_FIRINGS, NEIGHBORS_TO_FIRINGS, FIRING_STRUCTURES])
  })

  test("two identical searches return deeply equal results (determinism)", async () => {
    const provider = new BonfiresCxnProvider(CFG, ARTIFACTS, fakeDeps([]))
    const a = await provider.search("puppy?", { containerTag: "x" })
    const b = await provider.search("puppy?", { containerTag: "x" })
    expect(JSON.stringify(a)).toBe(JSON.stringify(b))
  })
})

describe("cypher templates", () => {
  test("every node pattern is group-scoped — no anonymous ungrouped endpoints", () => {
    for (const template of [SEED_ENTITIES, SEEDS_TO_FIRINGS, NEIGHBORS_TO_FIRINGS, FIRING_STRUCTURES]) {
      expect(template).not.toMatch(/-\(\)/)          // no bare anonymous endpoint
      expect(template).not.toMatch(/\(\)-/)
    }
    expect(SEED_ENTITIES).toContain("(n)-[r]-(m:Entity {group_id: $groupId})")
  })
})
