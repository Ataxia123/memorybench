// One-off: embed blip captions as first-class retrievable items (leg 5, lane B2).
// Usage: bun run scripts/cxn-embed-captions.ts --turns <session_turns.json> --out <captions_embeddings.json>
import { readFileSync, writeFileSync } from "node:fs"
import { embedTexts, VOYAGE_DIM, VOYAGE_MODEL } from "../src/providers/bonfires-cxn/voyage"

function arg(flag: string): string {
  const index = process.argv.indexOf(flag)
  if (index < 0 || !process.argv[index + 1]) throw new Error(`missing ${flag}`)
  return process.argv[index + 1]!
}

const apiKey = process.env.VOYAGE_API_KEY?.trim()
if (!apiKey) throw new Error("VOYAGE_API_KEY required")

interface Turn { ts: string; speaker: string; text: string; blip_caption: string | null }
const sessions = JSON.parse(readFileSync(arg("--turns"), "utf-8")) as Record<string, Turn[]>

const entries: Array<{ id: string; caption: string; ts: string; actor_id: string; session: string }> = []
for (const [session, turns] of Object.entries(sessions)) {
  for (const turn of turns) {
    if (!turn.blip_caption) continue
    const key = ["caption", session, turn.ts, turn.speaker, turn.blip_caption].join("\x00")
    const id = new Bun.CryptoHasher("sha256").update(key).digest("hex").slice(0, 16)
    entries.push({ id, caption: turn.blip_caption, ts: turn.ts, actor_id: turn.speaker, session })
  }
}
entries.sort((a, b) => (a.id < b.id ? -1 : 1))
const vectors = await embedTexts(entries.map((e) => `[image] ${e.caption}`), "document", apiKey)
if (vectors.some((v) => v.length !== VOYAGE_DIM)) throw new Error("unexpected embedding dim")

const items: Record<string, unknown> = {}
entries.forEach((entry, i) => {
  if (items[entry.id]) throw new Error(`duplicate caption id ${entry.id}`)
  items[entry.id] = { caption: entry.caption, ts: entry.ts, actor_id: entry.actor_id, session: entry.session, vector: vectors[i]! }
})
writeFileSync(arg("--out"), JSON.stringify({ model: VOYAGE_MODEL, dim: VOYAGE_DIM, items }, null, 1) + "\n")
console.log(`embedded ${entries.length} captions -> ${arg("--out")}`)
