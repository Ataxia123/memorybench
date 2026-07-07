// Live gate: real preflight against gate-B + byte-identical two-run search determinism.
// Controller-run with CXN_* env set. NOT part of `bun test` — invoked explicitly.
import { BonfiresCxnProvider } from "../src/providers/bonfires-cxn/index"

const QUESTIONS = [
  "When did Caroline adopt the puppy?",
  "How does Melanie feel about her family?",
  "What did Caroline say about support?",
  "Where did Melanie go camping?",
  "What compliment did Caroline give?",
]

async function main(): Promise<number> {
  const provider = new BonfiresCxnProvider()
  try {
    await provider.initialize({ apiKey: "none" })
  } catch (error) {
    console.error(`LIVE CHECK FAIL (initialize/preflight): ${String(error)}`)
    return 1
  }
  for (const question of QUESTIONS) {
    const first = JSON.stringify(await provider.search(question, { containerTag: "live-check" }))
    const second = JSON.stringify(await provider.search(question, { containerTag: "live-check" }))
    if (first !== second) {
      console.error(`LIVE CHECK FAIL (determinism): "${question}"`)
      return 1
    }
    const count = (JSON.parse(first) as unknown[]).length
    console.log(`ok: "${question}" -> ${count} result items (identical across 2 runs)`)
  }
  console.log("LIVE CHECK PASS")
  return 0
}

main().then((code) => process.exit(code))
