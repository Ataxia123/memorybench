import { describe, expect, test } from "bun:test"
import { BonfiresCxnProvider, type CxnDeps } from "./index"
import type { CxnArtifacts } from "./artifacts"
import type { CxnConfig } from "./config"

function config(overrides: Partial<CxnConfig> = {}): CxnConfig {
  return {
    neo4jUri: "bolt://x", neo4jUser: "u", neo4jPassword: "p",
    groupId: "g", artifactsDir: "/nowhere", utteranceMapPath: "/nowhere/map.json",
    expectedNodes: 2, expectedEdges: 1, topKFirings: 20, maxSeedEntities: 12,
    ...overrides,
  }
}

function artifacts(): CxnArtifacts {
  return {
    utteranceMap: new Map([["h1", { utterance: "hi", ts: "t", actor_id: "a", session: "s" }]]),
    entrenchmentByConstruct: new Map(),
    planRecordCount: 1,
  }
}

// Fake bolt: answers the three preflight queries in order of shape.
function fakeDeps(counts: { nodes: number; edges: number; firings: number }): CxnDeps {
  return {
    runCypher: async (query: string) => {
      if (query.includes("()-[r]-") || query.includes("-[r]->")) return [{ n: counts.edges }]
      if (query.includes("Entity_Firing")) {
        if (query.includes("utterance_hash") || query.includes("attributes"))
          return [{ uuid: "f1", attributes: JSON.stringify({ utterance_hash: "h1" }) }]
        return [{ n: counts.firings }]
      }
      return [{ n: counts.nodes }]
    },
  }
}

describe("preflight", () => {
  test("passes when counts and sampled hashes line up", async () => {
    const provider = new BonfiresCxnProvider(config(), artifacts(), fakeDeps({ nodes: 2, edges: 1, firings: 1 }))
    await provider.preflight()
  })

  test("hard-fails on node-count mismatch", async () => {
    const provider = new BonfiresCxnProvider(config(), artifacts(), fakeDeps({ nodes: 99, edges: 1, firings: 1 }))
    await expect(provider.preflight()).rejects.toThrow(/expectedNodes/)
  })

  test("hard-fails when a sampled firing hash is not in the sidecar map", async () => {
    const provider = new BonfiresCxnProvider(
      config(),
      { ...artifacts(), utteranceMap: new Map() },
      fakeDeps({ nodes: 2, edges: 1, firings: 1 })
    )
    await expect(provider.preflight()).rejects.toThrow(/sidecar/)
  })

  test("clear refuses to touch the KG", async () => {
    const forbidden: CxnDeps = {
      runCypher: async () => {
        throw new Error("clear() must never run Cypher")
      },
    }
    const provider = new BonfiresCxnProvider(config(), artifacts(), forbidden)
    await provider.clear("anything") // resolves ONLY if clear never touches the deps
  })
})
