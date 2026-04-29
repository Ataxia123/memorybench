import type { SearchHit } from "./search.js"

interface RerankOptions {
  /** Delve base URL — defaults to BONFIRES_API_URL or env. */
  baseUrl?: string
  topN: number
}

interface RerankResponseItem {
  index: number
  score: number
}

interface RerankResponseBody {
  results: RerankResponseItem[]
}

/** Final-stage cross-encoder rerank over the union of facts+entities+chunks.
 *
 * Posts to delve's ``/rerank`` route, which exposes the same BGE/Voyage
 * cross-encoder the chunks_search lane uses internally. A single rerank
 * pass over the merged pool — judged by a CE that scores query↔passage
 * pairs directly — demotes "topical but answer-wrong" hits across lane
 * boundaries.
 *
 * Failure mode: on HTTP error or parse failure we return the input list
 * truncated to ``topN``. Never throws — bench-only feature.
 */
export async function ceRerank(
  query: string,
  hits: SearchHit[],
  opts: RerankOptions
): Promise<SearchHit[]> {
  if (hits.length === 0) return hits
  if (hits.length <= opts.topN) return hits

  const baseUrl = opts.baseUrl ?? process.env.BONFIRES_API_URL ?? "http://localhost:8000"
  const url = `${baseUrl.replace(/\/+$/, "")}/rerank`

  // Cross-encoders score (query, passage) pairs as plain text — strip the
  // kind tag (entity/fact/chunk) so the CE doesn't pick up structural cues
  // and can compare passages on substantive content alone.
  const passages = hits.map((h) => h.text)

  let parsed: RerankResponseBody
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, passages, top_n: opts.topN }),
    })
    if (!res.ok) {
      console.warn(`ceRerank: delve /rerank returned ${res.status}, returning input order`)
      return hits.slice(0, opts.topN)
    }
    parsed = (await res.json()) as RerankResponseBody
  } catch (err) {
    console.warn("ceRerank: fetch failed, returning input order:", err)
    return hits.slice(0, opts.topN)
  }

  if (!Array.isArray(parsed.results)) {
    console.warn("ceRerank: malformed response, returning input order")
    return hits.slice(0, opts.topN)
  }

  const out: SearchHit[] = []
  for (const item of parsed.results) {
    const h = hits[item.index]
    if (h !== undefined) out.push(h)
    if (out.length >= opts.topN) break
  }
  return out
}
