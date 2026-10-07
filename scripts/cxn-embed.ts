// One-off: embed the statement corpus + compute cxn/episode aggregates.
// Controller-run with VOYAGE_API_KEY set. ~1091 texts ≈ $0.05.
// Usage: bun run scripts/cxn-embed.ts --corpus <statement_corpus.json> --out <embeddings.json>
import { readFileSync, writeFileSync } from "node:fs"
import { embedTexts, meanNormalized, VOYAGE_DIM, VOYAGE_MODEL } from "../src/providers/bonfires-cxn/voyage"

function arg(flag: string): string {
  const index = process.argv.indexOf(flag)
  if (index < 0 || !process.argv[index + 1]) throw new Error(`missing ${flag}`)
  return process.argv[index + 1]!
}

const apiKey = process.env.VOYAGE_API_KEY?.trim()
if (!apiKey) throw new Error("VOYAGE_API_KEY required")

const corpus = JSON.parse(readFileSync(arg("--corpus"), "utf-8")) as Record<
  string,
  { utterance: string; session: string; construct_ids: string[] }
>
const hashes = Object.keys(corpus).sort()
const vectors = await embedTexts(hashes.map((h) => corpus[h]!.utterance), "document", apiKey)
if (vectors.some((v) => v.length !== VOYAGE_DIM)) throw new Error("unexpected embedding dim")

const statements: Record<string, number[]> = {}
hashes.forEach((hash, i) => { statements[hash] = vectors[i]! })

const byCxn = new Map<string, number[][]>()
const byEpisode = new Map<string, number[][]>()
hashes.forEach((hash, i) => {
  const entry = corpus[hash]!
  for (const cid of entry.construct_ids) {
    if (cid === "residual.v1") continue                    // spec: monolith excluded
    byCxn.set(cid, [...(byCxn.get(cid) ?? []), vectors[i]!])
  }
  byEpisode.set(entry.session, [...(byEpisode.get(entry.session) ?? []), vectors[i]!])
})

const aggregates = {
  cxn: Object.fromEntries([...byCxn.entries()].sort().map(([k, v]) => [k, meanNormalized(v)])),
  episode: Object.fromEntries([...byEpisode.entries()].sort().map(([k, v]) => [k, meanNormalized(v)])),
}

writeFileSync(arg("--out"), JSON.stringify({ model: VOYAGE_MODEL, dim: VOYAGE_DIM, statements, aggregates }, null, 1))
console.error(`wrote embeddings: ${hashes.length} statements, ${byCxn.size} cxn + ${byEpisode.size} episode aggregates`)
