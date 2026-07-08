import { describe, expect, test } from "bun:test"
import { BonfiresKernelProvider, mapSearchEnvelope, type FetchLike } from "./index"
import { loadKernelConfig, type KernelConfig } from "./config"
import type { UnifiedSession } from "../../types/unified"

describe("bonfires-kernel provider mapping", () => {
  test("envelope maps to utterances + one context item carrying the directive", () => {
    const items = mapSearchEnvelope({
      results: [
        {
          uuid: "h1",
          score: 0.9,
          text: "[2023-05-01 10:00 melanie] x",
          metadata: { utterance_hash: "h1" },
        },
      ],
      context_lines: ["[2023-05-01 10:00 melanie] x"],
      directive: "P\nD",
      recipe: { directiveVersion: 2 },
      fallback: false,
    })
    const utterances = items.filter((i) => (i as { kind: string }).kind === "cxn_utterance")
    const context = items.filter((i) => (i as { kind: string }).kind === "cxn_context")
    expect(utterances.length).toBe(1)
    expect(context.length).toBe(1)
    expect((context[0] as { directive?: string }).directive).toBe("P\nD")
  })

  test("null directive yields context item without directive", () => {
    const items = mapSearchEnvelope({
      results: [],
      context_lines: [],
      directive: null,
      recipe: null,
      fallback: true,
    })
    const context = items.filter((i) => (i as { kind: string }).kind === "cxn_context")
    expect((context[0] as { directive?: string }).directive).toBeUndefined()
  })

  test("utterance items carry metadata verbatim from the hit", () => {
    const items = mapSearchEnvelope({
      results: [{ uuid: "h2", score: 0.5, text: "hi", metadata: { foo: "bar" } }],
      context_lines: [],
      directive: null,
      recipe: null,
      fallback: false,
    })
    expect(items[0]).toEqual({
      text: "hi",
      kind: "cxn_utterance",
      score: 0.5,
      metadata: { foo: "bar" },
    })
  })
})

describe("loadKernelConfig", () => {
  function baseEnv(
    overrides: Record<string, string | undefined> = {}
  ): Record<string, string | undefined> {
    return {
      KERNELB_API_URL: "http://localhost:8000",
      KERNELB_BONFIRE_ID: "bf1",
      KERNELB_API_KEY: "k1",
      ...overrides,
    }
  }

  test("required env missing throws", () => {
    expect(() => loadKernelConfig({})).toThrow(/KERNELB_API_URL/)
  })

  test("defaults actorId to bench and leaves expectedCardsDigest unset", () => {
    const cfg = loadKernelConfig(baseEnv())
    expect(cfg.actorId).toBe("bench")
    expect(cfg.expectedCardsDigest).toBeUndefined()
  })

  test("picks up actorId and expectedCardsDigest overrides", () => {
    const cfg = loadKernelConfig(
      baseEnv({ KERNELB_ACTOR_ID: "custom", KERNELB_EXPECTED_CARDS_DIGEST: "abc123" })
    )
    expect(cfg.actorId).toBe("custom")
    expect(cfg.expectedCardsDigest).toBe("abc123")
  })
})

function config(overrides: Partial<KernelConfig> = {}): KernelConfig {
  return {
    apiUrl: "http://localhost:8000",
    bonfireId: "bf1",
    apiKey: "k1",
    actorId: "bench",
    ...overrides,
  }
}

function session(id: string, messages: UnifiedSession["messages"]): UnifiedSession {
  return { sessionId: id, messages, metadata: { date: "2023-05-08" } }
}

describe("preflight (KERNELB_EXPECTED_CARDS_DIGEST)", () => {
  test("passes when GET /kernel/state digest matches", async () => {
    const fetchImpl: FetchLike = async () =>
      new Response(JSON.stringify({ recipe: { cards_digest: "abc123" } }), { status: 200 })
    const provider = new BonfiresKernelProvider(
      config({ expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    await provider.initialize({ apiKey: "k1" })
  })

  test("throws on digest mismatch", async () => {
    const fetchImpl: FetchLike = async () =>
      new Response(JSON.stringify({ recipe: { cards_digest: "other" } }), { status: 200 })
    const provider = new BonfiresKernelProvider(
      config({ expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    await expect(provider.initialize({ apiKey: "k1" })).rejects.toThrow(/cards_digest mismatch/)
  })

  test("skips preflight when no digest is configured", async () => {
    const fetchImpl: FetchLike = async () => {
      throw new Error("must not fetch when no expectedCardsDigest is set")
    }
    const provider = new BonfiresKernelProvider(config(), fetchImpl)
    await provider.initialize({ apiKey: "k1" })
  })
})

describe("ingest + awaitIndexing (cxn fold)", () => {
  test("POSTs message_batches once with metadata.cxn_fold=true, in text/username/timestamp shape", async () => {
    const calls: Array<{ url: string; headers: Record<string, string>; body: unknown }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      calls.push({
        url,
        headers: (init?.headers as Record<string, string>) || {},
        body: init?.body ? JSON.parse(init.body as string) : undefined,
      })
      return new Response(
        JSON.stringify({ census_digest: "d1", statement_count: 2, construct_universe: [] }),
        { status: 200 }
      )
    }
    const provider = new BonfiresKernelProvider(config(), fetchImpl)
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    expect(result.documentIds).toEqual(["s1"])

    // First awaitIndexing call triggers the fold POST.
    let progress: { completedIds: string[]; failedIds: string[]; total: number } | undefined
    await provider.awaitIndexing(result, "q1", (p) => {
      progress = p
    })
    expect(calls.length).toBe(1)
    expect(calls[0]?.url).toContain("/bonfires/bf1/kernel/index")
    expect(calls[0]?.headers["X-Permission"]).toBe("write")
    const body = calls[0]?.body as {
      actor_id: string
      mode: string
      metadata: { cxn_fold: boolean }
      message_batches: Array<
        Array<{
          text: string
          username: string
          timestamp: string
          metadata: Record<string, unknown>
        }>
      >
    }
    expect(body.actor_id).toBe("bench")
    expect(body.mode).toBe("upsert")
    expect(body.metadata).toEqual({ cxn_fold: true })
    expect(body.message_batches).toEqual([
      [
        {
          text: "hi there",
          username: "Melanie",
          timestamp: "2023-05-08T10:00:00Z",
          metadata: { session_id: "s1" },
        },
      ],
    ])
    expect(progress).toEqual({ completedIds: ["s1"], failedIds: [], total: 1 })

    // Second awaitIndexing call (another question sharing the same corpus)
    // must NOT re-POST.
    await provider.awaitIndexing(result, "q2")
    expect(calls.length).toBe(1)
  })
})

describe("search (thin HTTP mapping)", () => {
  test("POSTs {query, top_k} and maps the hits/context envelope", async () => {
    let capturedBody: unknown
    const fetchImpl: FetchLike = async (url, init) => {
      capturedBody = init?.body ? JSON.parse(init.body as string) : undefined
      expect(url).toContain("/bonfires/bf1/kernel/search")
      return new Response(
        JSON.stringify({
          hits: [{ uuid: "h1", score: 0.8, text: "hello", metadata: { m: 1 } }],
          context_lines: ["hello"],
          directive: "P\nD",
          recipe: { directiveVersion: 2 },
          fallback: false,
        }),
        { status: 200 }
      )
    }
    const provider = new BonfiresKernelProvider(config(), fetchImpl)
    const results = await provider.search("what happened?", { containerTag: "q1" })
    expect(capturedBody).toEqual({ query: "what happened?", top_k: 20 })
    expect(results.length).toBe(2)
    expect(results[0]).toEqual({
      text: "hello",
      kind: "cxn_utterance",
      score: 0.8,
      metadata: { m: 1 },
    })
    expect((results[1] as { directive?: string }).directive).toBe("P\nD")
    const diagnostics = (results as unknown as { diagnostics?: { recipe: unknown } }).diagnostics
    expect(diagnostics?.recipe).toEqual({ directiveVersion: 2 })
  })
})

describe("clear", () => {
  test("refuses to delete the fold bonfire", async () => {
    const fetchImpl: FetchLike = async () => {
      throw new Error("clear() must never call fetch")
    }
    const provider = new BonfiresKernelProvider(config(), fetchImpl)
    await provider.clear("anything")
  })
})
