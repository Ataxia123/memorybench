// Live gate: real preflight against gate-B + byte-identical two-run search
// determinism, under BOTH CXN_LANE_P states. Controller-run with CXN_* env
// set. NOT part of `bun test` — invoked explicitly.
import { BonfiresCxnProvider } from "../src/providers/bonfires-cxn/index"

const QUESTIONS = [
  "When did Caroline adopt the puppy?",
  "How does Melanie feel about her family?",
  "What did Caroline say about support?",
  "Where did Melanie go camping?",
  "What compliment did Caroline give?",
]

async function runState(laneP: string): Promise<boolean> {
  const previous = process.env.CXN_LANE_P
  process.env.CXN_LANE_P = laneP
  const provider = new BonfiresCxnProvider()
  try {
    await provider.initialize({ apiKey: "none" })
  } catch (error) {
    console.error(`LIVE CHECK FAIL (CXN_LANE_P=${laneP} initialize/preflight): ${String(error)}`)
    process.env.CXN_LANE_P = previous
    return false
  }

  let ok = true
  for (const question of QUESTIONS) {
    const first = JSON.stringify(await provider.search(question, { containerTag: "live-check" }))
    const second = JSON.stringify(await provider.search(question, { containerTag: "live-check" }))
    if (first !== second) {
      console.error(`LIVE CHECK FAIL (CXN_LANE_P=${laneP} determinism): "${question}"`)
      ok = false
      continue
    }
    const count = (JSON.parse(first) as unknown[]).length
    console.log(`ok (CXN_LANE_P=${laneP}): "${question}" -> ${count} result items (identical across 2 runs)`)
  }
  process.env.CXN_LANE_P = previous
  return ok
}

async function main(): Promise<number> {
  const floorOk = await runState("0")
  const laneOk = await runState("1")
  if (!floorOk || !laneOk) {
    console.error("LIVE CHECK FAIL")
    return 1
  }
  console.log("LIVE CHECK PASS (both CXN_LANE_P=0 and CXN_LANE_P=1 states deterministic)")
  return 0
}

main().then((code) => process.exit(code))
