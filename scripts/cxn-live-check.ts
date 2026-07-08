// Live gate: real preflight against gate-B + byte-identical two-run search
// determinism, under BOTH CXN_LANE_P states (leg 2) AND BOTH CXN_Q states
// (leg 3 — kernel-native comprehend sidecar), plus a live comprehend-sidecar
// idempotency check and a known-fallback-question ordering check. Controller-
// run with CXN_* env set (including CXN_COMPREHEND_URL for the CXN_Q=1 and
// comprehend checks). NOT part of `bun test` — invoked explicitly.
import { BonfiresCxnProvider } from "../src/providers/bonfires-cxn/index"

const QUESTIONS = [
  "When did Caroline adopt the puppy?",
  "How does Melanie feel about her family?",
  "What did Caroline say about support?",
  "Where did Melanie go camping?",
  "What compliment did Caroline give?",
]

// Fixed question for the N=3 POST /comprehend byte-identity check.
const COMPREHEND_QUESTION = "When did Caroline adopt the puppy?"

// A question the comprehend sidecar is expected to fail to strata-match
// against any real construct id, forcing the floor fallback
// (recipe.fallback === true — see src/providers/bonfires-cxn/affordance.ts's
// isFallback()).
const FALLBACK_QUESTION = "zzz qqq unmatched gibberish?"

async function runLaneState(laneP: string): Promise<boolean> {
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

// v3: the same 5-question x 2-run byte-identity gate as runLaneState, but
// toggling CXN_Q instead of CXN_LANE_P — an orthogonal axis. CXN_Q=1 requires
// CXN_COMPREHEND_URL to already be set in the environment (loadCxnConfig()
// throws otherwise); the controller is responsible for pointing that at a
// live sidecar before invoking this script.
async function runQState(cxnQ: string): Promise<boolean> {
  const previous = process.env.CXN_Q
  process.env.CXN_Q = cxnQ
  const provider = new BonfiresCxnProvider()
  try {
    await provider.initialize({ apiKey: "none" })
  } catch (error) {
    console.error(`LIVE CHECK FAIL (CXN_Q=${cxnQ} initialize/preflight): ${String(error)}`)
    process.env.CXN_Q = previous
    return false
  }

  let ok = true
  for (const question of QUESTIONS) {
    const first = JSON.stringify(await provider.search(question, { containerTag: "live-check-q" }))
    const second = JSON.stringify(await provider.search(question, { containerTag: "live-check-q" }))
    if (first !== second) {
      console.error(`LIVE CHECK FAIL (CXN_Q=${cxnQ} determinism): "${question}"`)
      ok = false
      continue
    }
    const count = (JSON.parse(first) as unknown[]).length
    console.log(`ok (CXN_Q=${cxnQ}): "${question}" -> ${count} result items (identical across 2 runs)`)
  }
  process.env.CXN_Q = previous
  return ok
}

// v3: POST the same question to the live comprehend sidecar's /comprehend
// endpoint 3 times and assert every response body is byte-identical JSON —
// the sidecar must be a pure function of its input, not something with
// hidden per-call state (cache warmup, nondeterministic ordering, etc.).
async function checkComprehendIdempotency(): Promise<boolean> {
  const comprehendUrl = process.env.CXN_COMPREHEND_URL
  if (!comprehendUrl) {
    console.error("LIVE CHECK FAIL (comprehend idempotency): CXN_COMPREHEND_URL is not set")
    return false
  }

  const bodies: string[] = []
  for (let i = 0; i < 3; i++) {
    let response: Response
    try {
      response = await fetch(`${comprehendUrl}/comprehend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: COMPREHEND_QUESTION }),
      })
    } catch (error) {
      console.error(`LIVE CHECK FAIL (comprehend idempotency): POST /comprehend threw: ${String(error)}`)
      return false
    }
    if (!response.ok) {
      console.error(`LIVE CHECK FAIL (comprehend idempotency): POST /comprehend returned ${response.status}`)
      return false
    }
    bodies.push(JSON.stringify(await response.json()))
  }

  const allIdentical = bodies.every((body) => body === bodies[0])
  if (!allIdentical) {
    console.error(
      `LIVE CHECK FAIL (comprehend idempotency): 3 POST /comprehend bodies differ for "${COMPREHEND_QUESTION}"`
    )
    return false
  }
  console.log(`ok (comprehend idempotency): "${COMPREHEND_QUESTION}" -> byte-identical across 3 POSTs`)
  return true
}

// Best-effort ordering fingerprint of a search() result array: the sequence
// of utterance hashes for cxn_utterance items, falling back to `kind` for
// anything else (e.g. the trailing cxn_context item) — deliberately ignores
// scores/recipe/lanes metadata, which are expected to differ between
// CXN_Q=0 and CXN_Q=1 even when retrieval ordering itself is unchanged.
function resultOrdering(results: unknown[]): string[] {
  return results.map((item) => {
    if (!item || typeof item !== "object") return String(item)
    const record = item as Record<string, unknown>
    if (record.kind === "cxn_utterance") {
      const metadata = record.metadata as Record<string, unknown> | undefined
      return String(metadata?.utterance_hash ?? "")
    }
    return String(record.kind ?? "")
  })
}

// v3: a question expected to miss every real construct id in the sidecar's
// grammar, forcing recipe.fallback === true (see affordance.ts's
// isFallback()). Falling back must be silent to retrieval — the result
// ORDERING for this question under CXN_Q=1 must match its CXN_Q=0 baseline
// (only the recipe's affordance metadata should differ).
async function checkFallbackQuestion(): Promise<boolean> {
  const previousQ = process.env.CXN_Q

  process.env.CXN_Q = "0"
  const floorProvider = new BonfiresCxnProvider()
  try {
    await floorProvider.initialize({ apiKey: "none" })
  } catch (error) {
    console.error(`LIVE CHECK FAIL (fallback question, CXN_Q=0 initialize): ${String(error)}`)
    process.env.CXN_Q = previousQ
    return false
  }
  const floorResults = await floorProvider.search(FALLBACK_QUESTION, { containerTag: "live-check-fallback" })
  const floorOrdering = resultOrdering(floorResults)

  process.env.CXN_Q = "1"
  const qProvider = new BonfiresCxnProvider()
  try {
    await qProvider.initialize({ apiKey: "none" })
  } catch (error) {
    console.error(`LIVE CHECK FAIL (fallback question, CXN_Q=1 initialize): ${String(error)}`)
    process.env.CXN_Q = previousQ
    return false
  }
  const qResults = await qProvider.search(FALLBACK_QUESTION, { containerTag: "live-check-fallback" })
  process.env.CXN_Q = previousQ

  const contextItem = qResults.find(
    (item): item is Record<string, unknown> =>
      !!item && typeof item === "object" && (item as Record<string, unknown>).kind === "cxn_context"
  )
  const recipe = contextItem?.recipe as Record<string, unknown> | undefined
  if (recipe?.fallback !== true) {
    console.error(
      `LIVE CHECK FAIL (fallback question): expected recipe.fallback === true for "${FALLBACK_QUESTION}", got ${String(recipe?.fallback)}`
    )
    return false
  }

  const qOrdering = resultOrdering(qResults)
  if (JSON.stringify(qOrdering) !== JSON.stringify(floorOrdering)) {
    console.error(
      `LIVE CHECK FAIL (fallback question): result ordering differs between CXN_Q=0 and CXN_Q=1 for "${FALLBACK_QUESTION}"`
    )
    return false
  }

  console.log(`ok (fallback question): "${FALLBACK_QUESTION}" -> recipe.fallback=true, ordering identical to CXN_Q=0`)
  return true
}

async function main(): Promise<number> {
  const floorOk = await runLaneState("0")
  const laneOk = await runLaneState("1")
  const qFloorOk = await runQState("0")
  const qOnOk = await runQState("1")
  const comprehendOk = await checkComprehendIdempotency()
  const fallbackOk = await checkFallbackQuestion()

  if (!floorOk || !laneOk || !qFloorOk || !qOnOk || !comprehendOk || !fallbackOk) {
    console.error("LIVE CHECK FAIL")
    return 1
  }
  console.log(
    "LIVE CHECK PASS (CXN_LANE_P=0/1 determinism, CXN_Q=0/1 determinism, comprehend idempotency, fallback ordering all verified)"
  )
  return 0
}

main().then((code) => process.exit(code))
