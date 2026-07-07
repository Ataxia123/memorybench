import { describe, expect, test } from "bun:test"
import { SEED_ENTITIES, SEEDS_TO_FIRINGS, NEIGHBORS_TO_FIRINGS, FIRING_STRUCTURES } from "./cypher"

// The leg-1 Cypher walk (seed match -> firing walk -> structure hydrate) is no
// longer exercised by search() (superseded by the v2 live-query dense floor +
// lane P flow, see search2.test.ts) but the templates stay pinned here — they
// document the topology and a future leg-B reuses them.
describe("cypher templates", () => {
  test("every node pattern is group-scoped — no anonymous ungrouped endpoints", () => {
    for (const template of [SEED_ENTITIES, SEEDS_TO_FIRINGS, NEIGHBORS_TO_FIRINGS, FIRING_STRUCTURES]) {
      expect(template).not.toMatch(/-\(\)/) // no bare anonymous endpoint
      expect(template).not.toMatch(/\(\)-/)
    }
    expect(SEED_ENTITIES).toContain("(n)-[r]-(m:Entity {group_id: $groupId})")
  })
})
