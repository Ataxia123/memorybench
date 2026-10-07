import { describe, expect, test } from "bun:test"
import { cosine, embedTexts, meanNormalized } from "./voyage"

function fakeFetch(capture: { bodies: unknown[] }) {
  return async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body))
    capture.bodies.push(body)
    const data = (body.input as string[]).map((text, i) => ({
      embedding: [text.length, i, 1],     // deterministic per input
    }))
    return new Response(JSON.stringify({ data }), { status: 200 })
  }
}

describe("embedTexts", () => {
  test("batches at 128, preserves order, passes input_type", async () => {
    const capture = { bodies: [] as any[] }
    const texts = Array.from({ length: 130 }, (_, i) => `t${i}`)
    const vectors = await embedTexts(texts, "document", "key", fakeFetch(capture) as any)
    expect(vectors.length).toBe(130)
    expect(capture.bodies.length).toBe(2)
    expect(capture.bodies[0].input.length).toBe(128)
    expect(capture.bodies[0].input_type).toBe("document")
    expect(capture.bodies[0].model).toBe("voyage-3")
    expect(vectors[0]![0]).toBe(2)        // "t0".length
  })

  test("throws on non-200", async () => {
    const bad = async () => new Response("nope", { status: 429 })
    await expect(embedTexts(["x"], "query", "key", bad as any)).rejects.toThrow(/429/)
  })
})

describe("vector math", () => {
  test("meanNormalized returns unit-length mean", () => {
    const m = meanNormalized([[1, 0], [0, 1]])
    const len = Math.hypot(...m)
    expect(Math.abs(len - 1)).toBeLessThan(1e-9)
    expect(m[0]).toBeCloseTo(m[1]!)
  })

  test("cosine of identical vectors is 1", () => {
    expect(cosine([0.6, 0.8], [0.6, 0.8])).toBeCloseTo(1)
  })
})
