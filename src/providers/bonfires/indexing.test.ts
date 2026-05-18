import { afterEach, describe, expect, mock, test } from "bun:test"
import type { UnifiedSession } from "../../types/unified.js"
import { LoCoMoBenchmark } from "../../benchmarks/locomo/index.js"
import { runIndexingPipeline, stackMessagesForSession } from "./indexing.js"

const previousCaptionFlag = process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS
const previousStackV2 = process.env.BONFIRES_STACK_V2
const previousStackV2NoDoc = process.env.BONFIRES_STACK_V2_NO_DOC
const previousOntologyResynth = process.env.BONFIRES_ONTOLOGY_RESYNTH
const previousCascadeEmbeddings = process.env.BONFIRES_CASCADE_EMBEDDINGS
const previousTier4Seed = process.env.BONFIRES_TIER4_SEED
const previousSkipEpisodes = process.env.BONFIRES_SKIP_EPISODES
const previousSkipBuildGrammar = process.env.BONFIRES_SKIP_BUILD_GRAMMAR

afterEach(() => {
  restoreEnv("LOCOMO_INCLUDE_IMAGE_CAPTIONS", previousCaptionFlag)
  restoreEnv("BONFIRES_STACK_V2", previousStackV2)
  restoreEnv("BONFIRES_STACK_V2_NO_DOC", previousStackV2NoDoc)
  restoreEnv("BONFIRES_ONTOLOGY_RESYNTH", previousOntologyResynth)
  restoreEnv("BONFIRES_CASCADE_EMBEDDINGS", previousCascadeEmbeddings)
  restoreEnv("BONFIRES_TIER4_SEED", previousTier4Seed)
  restoreEnv("BONFIRES_SKIP_EPISODES", previousSkipEpisodes)
  restoreEnv("BONFIRES_SKIP_BUILD_GRAMMAR", previousSkipBuildGrammar)
})

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name]
    return
  }
  process.env[name] = value
}

describe("runIndexingPipeline", () => {
  test("keeps the legacy pre-stack setup chain ordered when enabled", async () => {
    process.env.BONFIRES_STACK_V2 = "0"
    process.env.BONFIRES_STACK_V2_NO_DOC = "0"
    process.env.BONFIRES_ONTOLOGY_RESYNTH = "0"
    process.env.BONFIRES_CASCADE_EMBEDDINGS = "0"
    process.env.BONFIRES_TIER4_SEED = "0"
    process.env.BONFIRES_SKIP_EPISODES = "1"
    process.env.BONFIRES_SKIP_BUILD_GRAMMAR = "0"

    const order: string[] = []
    const client = {
      startSummaries: mock(async () => {
        order.push("startSummaries")
        return { job_id: "sum1" }
      }),
      waitForJob: mock(async (_id: string, opts: { kind: string }) => {
        order.push(`wait:${opts.kind}`)
        return { state: "completed" as const }
      }),
      startTaxonomy: mock(async () => {
        order.push("startTaxonomy")
        return { job_id: "tax1" }
      }),
      startLabelChunks: mock(async () => {
        order.push("startLabelChunks")
        return { job_id: "labels1" }
      }),
      buildChunksGrammar: mock(async () => {
        order.push("buildChunksGrammar")
        return {}
      }),
      buildGrammar: mock(async () => {
        order.push("buildGrammar")
        return {}
      }),
      buildCommunities: mock(async () => {
        order.push("buildCommunities")
        return {}
      }),
      buildOntology: mock(async () => {
        order.push("buildOntology")
        return {}
      }),
      getOntology: mock(async () => {
        order.push("getOntology")
        return { entity_labels: [] }
      }),
      setOntology: mock(async () => {
        order.push("setOntology")
        return {}
      }),
    }

    await runIndexingPipeline({
      client: client as unknown as Parameters<typeof runIndexingPipeline>[0]["client"],
      agentId: "agent-1",
      bonfireId: "bf-1",
      sessions: [],
    })

    expect(order).toEqual([
      "startSummaries",
      "wait:summaries",
      "startTaxonomy",
      "wait:taxonomy",
      "startLabelChunks",
      "wait:label_chunks",
      "buildChunksGrammar",
      "buildGrammar",
      "getOntology",
      "setOntology",
      "buildCommunities",
      "buildOntology",
    ])
  })
})

describe("Bonfires stack indexing", () => {
  test("uses stable source ids for dialogue text and image sidecars", () => {
    const session: UnifiedSession = {
      sessionId: "conv-26-session_11",
      messages: [
        {
          role: "user",
          speaker: "Melanie",
          content: "What concert was it?",
          metadata: {
            source_message_id: "conv-26-session_11-m1",
            source_kind: "message",
          },
        },
        {
          role: "user",
          speaker: "Melanie",
          content: "a photo of a poster for a concert with a picture of a man",
          metadata: {
            source_kind: "image_context",
            source_message_id: "conv-26-session_11-m1",
          },
        },
        {
          role: "user",
          speaker: "Melanie",
          content: "It was Matt Patterson.",
          metadata: {
            source_message_id: "conv-26-session_11-m2",
            source_kind: "message",
          },
        },
      ],
    }

    const messages = stackMessagesForSession(session, "salt")

    expect(messages.map((m) => m.id)).toEqual([
      "conv-26-session_11-m1",
      "conv-26-session_11-m1-image",
      "conv-26-session_11-m2",
    ])
    expect(messages[1].metadata?.source_message_id).toBe("conv-26-session_11-m1")
    expect(messages[1].metadata?.source_kind).toBe("image_context")
    expect(messages[2].text).toBe("It was Matt Patterson.")
  })

  test("does not double-suffix standalone image source ids", () => {
    const session: UnifiedSession = {
      sessionId: "conv-26-session_8",
      messages: [
        {
          role: "user",
          speaker: "Melanie",
          content: "a photo of a cup with a dog face on it",
          metadata: {
            source_kind: "image_context",
            source_message_id: "conv-26-session_8-m3-image",
          },
        },
      ],
    }

    const messages = stackMessagesForSession(session, "salt")

    expect(messages[0].id).toBe("conv-26-session_8-m3-image")
    expect(messages[0].metadata?.source_message_id).toBe("conv-26-session_8-m3-image")
  })
})

describe("LoCoMo image caption ingestion", () => {
  test("includes image captions by default", async () => {
    const previous = process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS
    delete process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS
    try {
      const benchmark = new LoCoMoBenchmark()
      await benchmark.load()

      const sessions = benchmark.getHaystackSessions("conv-26-q121")
      const session11 = sessions.find((s) => s.sessionId === "conv-26-session_11")
      expect(session11).toBeTruthy()

      const captionMessage = session11!.messages.find((m) =>
        m.content.includes("poster for a concert")
      )
      expect(captionMessage?.metadata?.source_kind).toBe("image_context")
      expect(captionMessage?.metadata?.source_message_id).toBe("conv-26-session_11-m1")
    } finally {
      if (previous === undefined) {
        delete process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS
      } else {
        process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS = previous
      }
    }
  })

  test("can disable image captions explicitly", async () => {
    const previous = process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS
    process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS = "0"
    try {
      const benchmark = new LoCoMoBenchmark()
      await benchmark.load()

      const sessions = benchmark.getHaystackSessions("conv-26-q121")
      const session11 = sessions.find((s) => s.sessionId === "conv-26-session_11")
      expect(session11).toBeTruthy()

      const captionMessage = session11!.messages.find((m) =>
        m.content.includes("poster for a concert")
      )
      expect(captionMessage).toBeUndefined()
    } finally {
      if (previous === undefined) {
        delete process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS
      } else {
        process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS = previous
      }
    }
  })

  test("keeps original dialogue source ids stable when captions are included", async () => {
    process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS = "1"
    const benchmark = new LoCoMoBenchmark()
    await benchmark.load()

    const sessions = benchmark.getHaystackSessions("conv-26-q121")
    const session11 = sessions.find((s) => s.sessionId === "conv-26-session_11")
    expect(session11).toBeTruthy()

    const mattMessage = session11!.messages.find((m) => m.content.includes("Matt Patterson"))
    expect(mattMessage?.metadata?.source_message_id).toBe("conv-26-session_11-m2")
    expect(mattMessage?.metadata?.source_kind).toBe("message")

    const captionMessage = session11!.messages.find((m) =>
      m.content.includes("poster for a concert")
    )
    expect(captionMessage?.metadata?.source_kind).toBe("image_context")
    expect(captionMessage?.metadata?.source_message_id).toBe("conv-26-session_11-m1")
  })

  test("does not emit empty dialogue messages when captions are included", async () => {
    process.env.LOCOMO_INCLUDE_IMAGE_CAPTIONS = "1"
    const benchmark = new LoCoMoBenchmark()
    await benchmark.load()

    const sessions = benchmark.getHaystackSessions("conv-26-q110")
    const session8 = sessions.find((s) => s.sessionId === "conv-26-session_8")
    expect(session8).toBeTruthy()

    const emptyMessages = session8!.messages.filter((m) => !m.content?.trim())
    expect(emptyMessages).toHaveLength(0)

    const captionMessage = session8!.messages.find((m) =>
      m.content.includes("cup with a dog face")
    )
    expect(captionMessage?.metadata?.source_kind).toBe("image_context")
    expect(captionMessage?.metadata?.source_message_id).toBe("conv-26-session_8-m3")
    expect(captionMessage?.metadata?.parent_source_message_id).toBe("conv-26-session_8-m3")
  })
})
