export const VOYAGE_MODEL = "voyage-3"
export const VOYAGE_DIM = 1024
const VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
const BATCH = 128

export type FetchLike = (url: string, init: RequestInit) => Promise<Response>

export async function embedTexts(
  texts: string[],
  inputType: "document" | "query",
  apiKey: string,
  fetchImpl: FetchLike = globalThis.fetch as FetchLike
): Promise<number[][]> {
  const out: number[][] = []
  for (let start = 0; start < texts.length; start += BATCH) {
    const input = texts.slice(start, start + BATCH)
    const response = await fetchImpl(VOYAGE_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ input, model: VOYAGE_MODEL, input_type: inputType }),
    })
    if (!response.ok) {
      throw new Error(`voyage embed failed: HTTP ${response.status} ${await response.text()}`)
    }
    const payload = (await response.json()) as { data: Array<{ embedding: number[] }> }
    if (!payload.data || payload.data.length !== input.length) {
      throw new Error(`voyage embed count mismatch: sent ${input.length}, got ${payload.data?.length}`)
    }
    for (const item of payload.data) out.push(item.embedding)
  }
  return out
}

export function meanNormalized(vectors: number[][]): number[] {
  if (vectors.length === 0) throw new Error("meanNormalized: empty input")
  const dim = vectors[0]!.length
  const sum = new Array<number>(dim).fill(0)
  for (const vector of vectors) for (let i = 0; i < dim; i++) sum[i]! += vector[i]!
  let norm = 0
  for (let i = 0; i < dim; i++) {
    sum[i]! /= vectors.length
    norm += sum[i]! * sum[i]!
  }
  norm = Math.sqrt(norm)
  if (norm === 0) throw new Error("meanNormalized: zero vector")
  return sum.map((value) => value / norm)
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!
    na += a[i]! * a[i]!
    nb += b[i]! * b[i]!
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}
