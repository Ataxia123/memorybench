import { describe, expect, test } from "bun:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
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

  test("leaves batchesPath and expectedCensusDigest unset by default", () => {
    const cfg = loadKernelConfig(baseEnv())
    expect(cfg.batchesPath).toBeUndefined()
    expect(cfg.expectedCensusDigest).toBeUndefined()
  })

  test("picks up KERNELB_BATCHES_PATH and KERNELB_EXPECTED_CENSUS_DIGEST overrides", () => {
    const cfg = loadKernelConfig(
      baseEnv({
        KERNELB_BATCHES_PATH: "/tmp/conv26_batches.json",
        KERNELB_EXPECTED_CENSUS_DIGEST: "digest123",
      })
    )
    expect(cfg.batchesPath).toBe("/tmp/conv26_batches.json")
    expect(cfg.expectedCensusDigest).toBe("digest123")
  })

  test("defaults skipFold to false", () => {
    const cfg = loadKernelConfig(baseEnv())
    expect(cfg.skipFold).toBe(false)
  })

  test.each(["1", "true", "TRUE"])("KERNELB_SKIP_FOLD=%s parses to true", (value) => {
    const cfg = loadKernelConfig(baseEnv({ KERNELB_SKIP_FOLD: value }))
    expect(cfg.skipFold).toBe(true)
  })

  test.each(["0", "false", "FALSE"])("KERNELB_SKIP_FOLD=%s parses to false", (value) => {
    const cfg = loadKernelConfig(baseEnv({ KERNELB_SKIP_FOLD: value }))
    expect(cfg.skipFold).toBe(false)
  })

  test("KERNELB_SKIP_FOLD with a non-bool value throws (strict bool)", () => {
    expect(() => loadKernelConfig(baseEnv({ KERNELB_SKIP_FOLD: "yes" }))).toThrow(
      /KERNELB_SKIP_FOLD must be one of/
    )
  })
})

function config(overrides: Partial<KernelConfig> = {}): KernelConfig {
  return {
    apiUrl: "http://localhost:8000",
    bonfireId: "bf1",
    apiKey: "k1",
    actorId: "bench",
    skipFold: false,
    ...overrides,
  }
}

function session(id: string, messages: UnifiedSession["messages"]): UnifiedSession {
  return { sessionId: id, messages, metadata: { date: "2023-05-08" } }
}

describe("preflight (KERNELB_EXPECTED_CARDS_DIGEST) — runs post-fold, not in initialize()", () => {
  function makeFoldAndStateFetch(
    cardsDigest: string,
    calls: Array<{ method: string; url: string }>
  ): FetchLike {
    return async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/index")) {
        return new Response(
          JSON.stringify({ census_digest: "d1", statement_count: 1, construct_universe: [] }),
          { status: 200 }
        )
      }
      if (url.includes("/kernel/state")) {
        return new Response(JSON.stringify({ recipe: { cards_digest: cardsDigest } }), {
          status: 200,
        })
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
  }

  test("initialize() never calls fetch — a fresh bonfire has no census yet and GET /kernel/state 503s pre-fold", async () => {
    const fetchImpl: FetchLike = async () => {
      throw new Error("initialize() must not fetch")
    }
    const provider = new BonfiresKernelProvider(
      config({ expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    await provider.initialize({ apiKey: "k1" })
  })

  test("passes when GET /kernel/state digest matches, called AFTER the fold POST (call order asserted)", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl = makeFoldAndStateFetch("abc123", calls)
    const provider = new BonfiresKernelProvider(
      config({ expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    await provider.initialize({ apiKey: "k1" })
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await provider.awaitIndexing(result, "q1")

    expect(calls.length).toBe(2)
    expect(calls[0]?.method).toBe("POST")
    expect(calls[0]?.url).toContain("/kernel/index")
    expect(calls[1]?.method).toBe("GET")
    expect(calls[1]?.url).toContain("/kernel/state")
  })

  test("throws on digest mismatch, discovered only after the fold POST has already succeeded", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl = makeFoldAndStateFetch("other", calls)
    const provider = new BonfiresKernelProvider(
      config({ expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    await provider.initialize({ apiKey: "k1" })
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).rejects.toThrow(/cards_digest mismatch/)
    // The fold POST already ran (and "succeeded") before the mismatch was discovered.
    expect(calls.length).toBe(2)
    expect(calls[0]?.method).toBe("POST")
    expect(calls[1]?.method).toBe("GET")
  })

  test("skips preflight (no GET /kernel/state at all) when no digest is configured", async () => {
    const fetchImpl: FetchLike = async (url) => {
      if (url.includes("/kernel/state")) {
        throw new Error("must not fetch /kernel/state when no expectedCardsDigest is set")
      }
      return new Response(
        JSON.stringify({ census_digest: "d1", statement_count: 1, construct_universe: [] }),
        { status: 200 }
      )
    }
    const provider = new BonfiresKernelProvider(config(), fetchImpl)
    await provider.initialize({ apiKey: "k1" })
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await provider.awaitIndexing(result, "q1")
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

  test("missing timestamp with no session metadata.date throws instead of inventing wall-clock time", async () => {
    const fetchImpl: FetchLike = async () => {
      throw new Error("must not fetch when sessionToMessageBatch throws first")
    }
    const provider = new BonfiresKernelProvider(config(), fetchImpl)
    const s1: UnifiedSession = {
      sessionId: "s1",
      messages: [{ role: "user", content: "hi there", speaker: "Melanie" }],
      // no metadata.date
    }
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).rejects.toThrow(
      /no timestamp and session\.metadata\.date is unset/
    )
  })
})

describe("pinned-batches mode (KERNELB_BATCHES_PATH)", () => {
  test("foldIndex reads the batches file and POSTs its content verbatim, bypassing sessionToMessageBatch", async () => {
    const pinnedBatches = [
      [
        {
          text: "Hey Mel! Good to see you!",
          username: "Caroline",
          timestamp: "2023-05-08T13:56:00+00:00",
          metadata: { sample_id: "conv-26", sessionId: "conv-26-session_1", dia_id: "D1:1" },
        },
      ],
    ]
    const dir = mkdtempSync(join(tmpdir(), "kernel-batches-"))
    const batchesPath = join(dir, "batches.json")
    writeFileSync(batchesPath, JSON.stringify(pinnedBatches))

    const calls: Array<{ body: unknown }> = []
    const fetchImpl: FetchLike = async (_url, init) => {
      calls.push({ body: init?.body ? JSON.parse(init.body as string) : undefined })
      return new Response(
        JSON.stringify({ census_digest: "d1", statement_count: 1, construct_universe: [] }),
        { status: 200 }
      )
    }
    const provider = new BonfiresKernelProvider(config({ batchesPath }), fetchImpl)

    // A session whose message has no timestamp AND no metadata.date — this
    // would THROW if sessionToMessageBatch ran on it, proving pinned-batches
    // mode bypasses reassembly entirely rather than merely overriding it.
    const s1: UnifiedSession = {
      sessionId: "s1",
      messages: [{ role: "user", content: "hi there", speaker: "Melanie" }],
    }
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await provider.awaitIndexing(result, "q1")

    expect(calls.length).toBe(1)
    const body = calls[0]?.body as { message_batches: unknown }
    expect(body.message_batches).toEqual(pinnedBatches)
  })
})

describe("census-digest tripwire (KERNELB_EXPECTED_CENSUS_DIGEST)", () => {
  function session(id: string, messages: UnifiedSession["messages"]): UnifiedSession {
    return { sessionId: id, messages, metadata: { date: "2023-05-08" } }
  }

  test("passes when the fold response census_digest matches", async () => {
    const fetchImpl: FetchLike = async () =>
      new Response(
        JSON.stringify({ census_digest: "expected-digest", statement_count: 1, construct_universe: [] }),
        { status: 200 }
      )
    const provider = new BonfiresKernelProvider(
      config({ expectedCensusDigest: "expected-digest" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).resolves.toBeUndefined()
  })

  test("throws with both digests named when the fold response census_digest mismatches", async () => {
    const fetchImpl: FetchLike = async () =>
      new Response(
        JSON.stringify({ census_digest: "drifted-digest", statement_count: 1, construct_universe: [] }),
        { status: 200 }
      )
    const provider = new BonfiresKernelProvider(
      config({ expectedCensusDigest: "expected-digest" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).rejects.toThrow(
      /census_digest mismatch — expected expected-digest, got drifted-digest/
    )
  })
})

describe("skip-fold mode (KERNELB_SKIP_FOLD) — search-only over a pre-folded bonfire", () => {
  test("issues zero index POSTs and no preflight when no digest is configured", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      calls.push({ method: init?.method ?? "GET", url })
      throw new Error(`unexpected fetch in test: ${url}`)
    }
    const provider = new BonfiresKernelProvider(config({ skipFold: true }), fetchImpl)
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await provider.awaitIndexing(result, "q1")
    expect(calls.length).toBe(0)

    // A second call (another question sharing the same corpus) must also
    // never fetch — the indexingDone guard applies to skip-fold mode too.
    await provider.awaitIndexing(result, "q2")
    expect(calls.length).toBe(0)
  })

  test("still preflights GET /kernel/state when KERNELB_EXPECTED_CARDS_DIGEST is set, and passes on match", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/state")) {
        return new Response(JSON.stringify({ recipe: { cards_digest: "abc123" } }), { status: 200 })
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
    const provider = new BonfiresKernelProvider(
      config({ skipFold: true, expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await provider.awaitIndexing(result, "q1")

    expect(calls.length).toBe(1)
    expect(calls[0]?.method).toBe("GET")
    expect(calls[0]?.url).toContain("/kernel/state")
  })

  test("digest mismatch throws — never POSTs /kernel/index even though it fails", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/state")) {
        return new Response(JSON.stringify({ recipe: { cards_digest: "other" } }), { status: 200 })
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
    const provider = new BonfiresKernelProvider(
      config({ skipFold: true, expectedCardsDigest: "abc123" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).rejects.toThrow(/cards_digest mismatch/)
    expect(calls.length).toBe(1)
    expect(calls.some((c) => c.url.includes("/kernel/index"))).toBe(false)
  })

  // I-2: the census tripwire (KERNELB_EXPECTED_CENSUS_DIGEST) only ran inside
  // foldIndex()'s fold-POST response check, so setting it in skip-fold mode
  // was previously a total no-op — the one HTTP call skip-fold makes (the
  // cards preflight GET) never looked at census_digest at all. These three
  // cases pin the fix: match passes, mismatch throws, and — unchanged from
  // before this fix — no digest configured means no preflight fetch at all.
  test("skip-fold + KERNELB_EXPECTED_CENSUS_DIGEST: passes when GET /kernel/state census_digest matches, via the same state fetch (single HTTP call)", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/state")) {
        return new Response(JSON.stringify({ census_digest: "expected-digest" }), { status: 200 })
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
    const provider = new BonfiresKernelProvider(
      config({ skipFold: true, expectedCensusDigest: "expected-digest" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).resolves.toBeUndefined()
    expect(calls.length).toBe(1)
    expect(calls[0]?.url).toContain("/kernel/state")
  })

  test("skip-fold + KERNELB_EXPECTED_CENSUS_DIGEST: mismatch throws — never POSTs /kernel/index", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/state")) {
        return new Response(JSON.stringify({ census_digest: "drifted-digest" }), { status: 200 })
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
    const provider = new BonfiresKernelProvider(
      config({ skipFold: true, expectedCensusDigest: "expected-digest" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).rejects.toThrow(
      /skip-fold census_digest mismatch — expected expected-digest, got drifted-digest/
    )
    expect(calls.length).toBe(1)
    expect(calls.some((c) => c.url.includes("/kernel/index"))).toBe(false)
  })

  test("skip-fold + KERNELB_EXPECTED_CENSUS_DIGEST: absent field throws (treated the same as a mismatch, not a silent pass)", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/state")) {
        return new Response(JSON.stringify({}), { status: 200 })
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
    const provider = new BonfiresKernelProvider(
      config({ skipFold: true, expectedCensusDigest: "expected-digest" }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).rejects.toThrow(
      /skip-fold census_digest mismatch — expected expected-digest, got \(absent from GET \/kernel\/state\)/
    )
    expect(calls.length).toBe(1)
  })

  test("skip-fold + both digests configured: a single shared GET /kernel/state backs both checks (no second HTTP call)", async () => {
    const calls: Array<{ method: string; url: string }> = []
    const fetchImpl: FetchLike = async (url, init) => {
      const method = init?.method ?? "GET"
      calls.push({ method, url })
      if (url.includes("/kernel/state")) {
        return new Response(
          JSON.stringify({ recipe: { cards_digest: "abc123" }, census_digest: "expected-digest" }),
          { status: 200 }
        )
      }
      throw new Error(`unexpected fetch in test: ${method} ${url}`)
    }
    const provider = new BonfiresKernelProvider(
      config({
        skipFold: true,
        expectedCardsDigest: "abc123",
        expectedCensusDigest: "expected-digest",
      }),
      fetchImpl
    )
    const s1 = session("s1", [
      { role: "user", content: "hi there", speaker: "Melanie", timestamp: "2023-05-08T10:00:00Z" },
    ])
    const result = await provider.ingest([s1], { containerTag: "q1" })
    await expect(provider.awaitIndexing(result, "q1")).resolves.toBeUndefined()
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
