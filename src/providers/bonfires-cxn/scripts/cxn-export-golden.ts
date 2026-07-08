// Golden-fixture exporter: captures the TS bonfires-cxn provider's per-stage
// retrieval outputs on 20 pinned conv-26 questions so the memory_kernel
// Python port can be verified against them byte-for-byte, without either
// side needing to call Voyage/the comprehend sidecar during CI.
//
// `buildGoldenEntry` (and its `toSparseTop40` helper) are PURE — stages in,
// JSON-shaped entry out — and are unit-tested without artifacts or network
// (export-golden.test.ts). `main()` does the heavy lifting: it loads the
// LoCoMo questions, loads cfg/artifacts independently of the provider,
// recomputes every stage with the SAME exported primitives the provider's
// own search() uses, ALSO runs the real provider.search() as a drift guard,
// and asserts the two agree before writing golden_parity.json. `main()` is
// guarded by `import.meta.main` so importing this module for unit tests never
// touches the network, Neo4j, or the filesystem outside the pinned test.
//
// Controller-run with the gate-B env (CXN_* pointed at the gate-B fold,
// CXN_Q=1 CXN_LANE_P=0 CXN_MMR=1 CXN_CAPTIONS=0, CXN_COMPREHEND_URL pointed
// at a live sidecar, CXN_VOYAGE_API_KEY set) — mirrors scripts/cxn-live-check.ts's
// instantiation mechanism (no-arg constructor + initialize()).
// Usage: bun run src/providers/bonfires-cxn/scripts/cxn-export-golden.ts --out <path>
import { readFile, writeFile } from "node:fs/promises"
import { loadCxnConfig, type CxnConfig } from "../config"
import { loadArtifacts } from "../artifacts"
import { embedTexts, type FetchLike } from "../voyage"
import {
  applyTemporalBoost,
  blendScores,
  bm25Scores,
  bm25ScoresWeighted,
  buildBm25,
  CORPUS_YEAR,
  denseScores,
  mmrSelect,
  replyExpansion,
  temporalWindow,
  type Bm25Index,
} from "../retrieval2"
import {
  answerDirective,
  combineSparse,
  directivePreamble,
  seededTermWeights,
  strataBoost,
  temporalWindowFromDates,
  type AffordanceKey,
  type Comprehension,
} from "../affordance"
import { BonfiresCxnProvider } from "../index"

// ---------- pinned question set ----------

// Pinned conv-26 question set (index N of conv-26's `qa` array = "conv-26-qN"):
// covers every affordance channel (q:seed / q:strata / q:temporal / q:answer)
// and every wh_slot shape (list, when, who/where/what, inference, etc).
export const GOLDEN_QUESTION_IDS: readonly string[] = [
  "conv-26-q0",
  "conv-26-q2",
  "conv-26-q5",
  "conv-26-q19",
  "conv-26-q26",
  "conv-26-q34",
  "conv-26-q52",
  "conv-26-q56",
  "conv-26-q60",
  "conv-26-q61",
  "conv-26-q66",
  "conv-26-q75",
  "conv-26-q78",
  "conv-26-q112",
  "conv-26-q124",
  "conv-26-q154",
  "conv-26-q183",
  "conv-26-q191",
  "conv-26-q196",
  "conv-26-q198",
] as const

// ---------- pure shape helpers (unit-testable, no artifacts/network) ----------

export interface GoldenStageInputs {
  qid: string
  question: string
  comprehension: Comprehension
  sparseTop40: Array<[string, number]>
  seededSparseTop40: Array<[string, number]>
  denseScores: Map<string, number>
  final20: string[]
  directive: string | null
  affordancesFired: string[]
  mmrFired: boolean
}

export interface GoldenEntry {
  qid: string
  question: string
  comprehension: Comprehension
  sparse_top40: Array<[string, number]>
  seeded_sparse_top40: Array<[string, number]>
  dense_scores: Record<string, number>
  final20: string[]
  directive: string | null
  affordances_fired: string[]
  mmr_fired: boolean
}

// Pure: stages in, JSON-shaped entry out. dense_scores is emitted at full
// precision with keys sorted ascending (stable diffing); sparse_top40 /
// seeded_sparse_top40 are passed through as already-rounded/sorted pairs
// (see toSparseTop40) — this function does no rounding or reordering of its
// own, it only reshapes.
export function buildGoldenEntry(stages: GoldenStageInputs): GoldenEntry {
  const dense_scores: Record<string, number> = {}
  for (const key of [...stages.denseScores.keys()].sort()) {
    dense_scores[key] = stages.denseScores.get(key)!
  }
  return {
    qid: stages.qid,
    question: stages.question,
    comprehension: stages.comprehension,
    sparse_top40: stages.sparseTop40,
    seeded_sparse_top40: stages.seededSparseTop40,
    dense_scores,
    final20: stages.final20,
    directive: stages.directive,
    affordances_fired: stages.affordancesFired,
    mmr_fired: stages.mmrFired,
  }
}

// "sparse_top40" (spec): the top N [hash, score] pairs of a sparse score map,
// sorted score-desc then hash-asc, scores rounded to 6 decimals.
export function toSparseTop40(scores: Map<string, number>, n = 40): Array<[string, number]> {
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, n)
    .map(([hash, score]): [string, number] => [hash, Math.round(score * 1e6) / 1e6])
}

// Recursively sorts object keys so JSON.stringify output is stable/diffable.
function sortKeysDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeysDeep)
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      out[key] = sortKeysDeep((value as Record<string, unknown>)[key])
    }
    return out
  }
  return value
}

function qidToIndex(qid: string): number {
  const match = /^conv-26-q(\d+)$/.exec(qid)
  if (!match) throw new Error(`cxn-export-golden: malformed qid "${qid}" (expected conv-26-q<N>)`)
  return Number(match[1])
}

function argValue(flag: string): string | undefined {
  const index = process.argv.indexOf(flag)
  if (index < 0) return undefined
  return process.argv[index + 1]
}

// Best-effort extraction of a search() result array's final utterance-hash
// set + trailing cxn_context directive — same shape live-check.ts's helpers
// read, duplicated locally rather than imported since those are private to
// the live-check script.
function extractProviderOutcome(results: unknown[]): {
  finalHashes: string[]
  directive: string | null
} {
  const finalHashes = results
    .map((item) => {
      if (!item || typeof item !== "object") return null
      const record = item as Record<string, unknown>
      if (record.kind !== "cxn_utterance") return null
      const metadata = record.metadata as Record<string, unknown> | undefined
      const id = metadata?.utterance_hash ?? metadata?.caption_id
      return typeof id === "string" ? id : null
    })
    .filter((id): id is string => id !== null)
  const contextItem = results.find(
    (item): item is Record<string, unknown> =>
      !!item && typeof item === "object" && (item as Record<string, unknown>).kind === "cxn_context"
  )
  const directive = (contextItem?.directive as string | null | undefined) ?? null
  return { finalHashes, directive }
}

async function main(): Promise<void> {
  const outPath = argValue("--out")
  if (!outPath) throw new Error("cxn-export-golden: --out <path> is required")

  const locomoRaw = JSON.parse(
    await readFile("data/benchmarks/locomo/locomo10.json", "utf-8")
  ) as Array<{
    sample_id: string
    qa: Array<{ question: string }>
  }>
  const conv26 = locomoRaw.find((item) => item.sample_id === "conv-26")
  if (!conv26) throw new Error("cxn-export-golden: conv-26 not found in locomo10.json")

  const questionsById = new Map<string, string>()
  for (const qid of GOLDEN_QUESTION_IDS) {
    const index = qidToIndex(qid)
    const qa = conv26.qa[index]
    if (!qa) throw new Error(`cxn-export-golden: no qa[${index}] for ${qid}`)
    questionsById.set(qid, qa.question)
  }

  // Independently loaded cfg/artifacts/BM25 — a SEPARATE code path from the
  // provider's own internal state, so the drift-guard assertion below means
  // something (two independent implementations agreeing), not a path
  // checked against itself.
  const cfg: CxnConfig = loadCxnConfig()
  const artifacts = await loadArtifacts(cfg)
  if (!artifacts.statements || !artifacts.vectors || !artifacts.aggregates) {
    throw new Error(
      "cxn-export-golden: artifacts missing v2 fields (statements/vectors/aggregates)"
    )
  }
  const statements = artifacts.statements
  const vectors = artifacts.vectors
  const bm25: Bm25Index = buildBm25([...statements.values()])

  const blendDense = cfg.blendDense ?? 0.7
  const blendSparse = cfg.blendSparse ?? 0.3
  const poolK = cfg.poolK ?? 40
  const finalK = cfg.finalK ?? 20
  const qSeed = cfg.qSeed ?? true
  const qStrata = cfg.qStrata ?? true
  const qGates = cfg.qGates ?? true
  const qAnswer = cfg.qAnswer ?? true
  const qDelta = cfg.qDelta ?? 0.3
  const qSeedEntityW = cfg.qSeedEntityW ?? 2.0
  const qSeedVerbW = cfg.qSeedVerbW ?? 1.0
  const qSeedLaneW = cfg.qSeedLaneW ?? 0.5
  const mmrOn = cfg.mmr ?? false
  const mmrLambda = cfg.mmrLambda ?? 0.3

  // Single caching fetch wrapper layered over the real fetch: guarantees
  // exactly ONE live network round trip per unique request (URL + body),
  // even though both this script's own embed/comprehend calls AND the
  // provider's internal embedQuery()/comprehendQuery() calls issue the
  // identical request for the same question. Response objects are cloned so
  // each caller gets its own readable body stream.
  const embedCache = new Map<string, Response>()
  const realFetch = globalThis.fetch.bind(globalThis)
  const cachingFetch: FetchLike = async (url, init) => {
    const key = `${url}::${String(init.body ?? "")}`
    const cached = embedCache.get(key)
    if (cached) return cached.clone()
    const response = await realFetch(url, init)
    embedCache.set(key, response.clone())
    return response
  }

  const originalFetch = globalThis.fetch
  // Monkey-patch global fetch for the script's duration so the provider's
  // internal `globalThis.fetch as FetchLike` fallback (used whenever
  // deps.fetchImpl is unset, which it is for the no-arg constructor path)
  // routes through the same caching wrapper this script's own embed/comprehend
  // calls use above.
  globalThis.fetch = cachingFetch as unknown as typeof fetch

  const entries: GoldenEntry[] = []
  try {
    // Mirrors scripts/cxn-live-check.ts's instantiation mechanism exactly:
    // no-arg constructor (env-driven config/artifacts/Neo4j deps) +
    // initialize({apiKey:"none"}).
    const provider = new BonfiresCxnProvider()
    await provider.initialize({ apiKey: "none" })

    for (const qid of GOLDEN_QUESTION_IDS) {
      const question = questionsById.get(qid)!

      // Live comprehend call — the same sidecar endpoint the provider itself
      // calls internally — hit directly so the fixture can carry the FULL
      // comprehension object (fillers/date_fillers/operators/bound_cxn_ids),
      // which the provider's own recipe.comprehend only echoes a subset of.
      const comprehendUrl = cfg.comprehendUrl
      if (!comprehendUrl) throw new Error("cxn-export-golden: CXN_COMPREHEND_URL is not set")
      const comprehendResponse = await cachingFetch(`${comprehendUrl}/comprehend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: question }),
      })
      if (!comprehendResponse.ok) {
        throw new Error(
          `cxn-export-golden: /comprehend failed (${comprehendResponse.status}) for ${qid}`
        )
      }
      const comprehension = (await comprehendResponse.json()) as Comprehension

      // Live query embed (one per question — cachingFetch makes the
      // provider's own internal embed call below reuse this response).
      const [queryVector] = await embedTexts([question], "query", cfg.voyageApiKey!, cachingFetch)
      if (!queryVector) throw new Error(`cxn-export-golden: embed returned no vector for ${qid}`)

      // ---- recompute stages with the SAME exported primitives the provider
      // uses (index.ts search(), lines ~330-483). Lane P and captions are OFF
      // in the export arm (CXN_LANE_P=0 CXN_CAPTIONS=0), so those two
      // branches are intentionally not recomputed here.
      const affordancesFired: AffordanceKey[] = []
      const naturalSparse = bm25Scores(bm25, question)
      let sparse = naturalSparse
      if (qSeed && comprehension.fillers.length) {
        const seeded = bm25ScoresWeighted(
          bm25,
          seededTermWeights(comprehension.fillers, qSeedEntityW, qSeedVerbW)
        )
        sparse = combineSparse(naturalSparse, seeded, qSeedLaneW)
        affordancesFired.push("q:seed")
      }

      const dense = denseScores(queryVector, vectors)
      let scores = blendScores(dense, sparse, blendDense, blendSparse)

      if (qStrata && comprehension.matched_cxn_ids.length) {
        scores = strataBoost(scores, statements, comprehension.matched_cxn_ids, qDelta)
        affordancesFired.push("q:strata")
      }

      const tsById: Map<string, { ts: string }> = new Map(statements)
      if (qGates) {
        const window = temporalWindowFromDates(comprehension.date_fillers, CORPUS_YEAR)
        if (window) {
          scores = applyTemporalBoost(scores, tsById, window, 0.15)
          affordancesFired.push("q:temporal")
        }
      } else {
        const window = temporalWindow(question)
        if (window) scores = applyTemporalBoost(scores, tsById, window, 0.15)
      }

      const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
      const pool = ranked.slice(0, poolK).map(([hash]) => hash)
      const poolScores = new Map(ranked.slice(0, poolK))
      const { added } = replyExpansion(pool, poolScores, statements, 10, 0.8)
      for (const { hash, score } of added) poolScores.set(hash, score)

      const mmrFired = mmrOn && comprehension.wh_slot === "list"
      const final = mmrFired
        ? mmrSelect(poolScores, (id) => vectors.get(id), finalK, mmrLambda)
        : [...poolScores.entries()]
            .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
            .slice(0, finalK)
      if (mmrFired) affordancesFired.push("b:mmr")
      const final20 = final.map(([hash]) => hash)

      const slotDirective = qAnswer ? answerDirective(comprehension.wh_slot) : null
      const directive = slotDirective ? `${directivePreamble()}\n${slotDirective}` : null
      if (directive) affordancesFired.push("q:answer")

      // ---- drift guard: run the real provider search() and assert the
      // recomputed final-20 is the EXACT SAME SET as the provider's returned
      // utterance hashes (order-insensitive set equality only — final's own
      // ordering is score/mmr order, the provider's is chronological), AND
      // that the recomputed directive matches the provider's cxn_context
      // directive exactly.
      const results = await provider.search(question, { containerTag: "golden-export" })
      const { finalHashes: providerFinalHashes, directive: providerDirective } =
        extractProviderOutcome(results)

      const recomputedSet = new Set(final20)
      const providerSet = new Set(providerFinalHashes)
      const setsEqual =
        recomputedSet.size === providerSet.size &&
        [...recomputedSet].every((id) => providerSet.has(id))
      if (!setsEqual) {
        throw new Error(
          `cxn-export-golden: FINAL-20 DRIFT for ${qid} — recomputed ${JSON.stringify([...recomputedSet].sort())} ` +
            `!= provider ${JSON.stringify([...providerSet].sort())}`
        )
      }
      if (directive !== providerDirective) {
        throw new Error(
          `cxn-export-golden: DIRECTIVE DRIFT for ${qid} — recomputed ${JSON.stringify(directive)} ` +
            `!= provider ${JSON.stringify(providerDirective)}`
        )
      }

      const sparseTop40 = toSparseTop40(naturalSparse)
      const seededSparseTop40 = toSparseTop40(sparse)

      entries.push(
        buildGoldenEntry({
          qid,
          question,
          comprehension,
          sparseTop40,
          seededSparseTop40,
          denseScores: dense,
          final20,
          directive,
          affordancesFired,
          mmrFired,
        })
      )
      console.log(`ok: ${qid} -> final-20 + directive parity verified against provider.search()`)
    }
  } finally {
    globalThis.fetch = originalFetch
  }

  const output = {
    meta: {
      corpus_year: CORPUS_YEAR,
      blend: [blendDense, blendSparse],
      pool_k: poolK,
      final_k: finalK,
    },
    questions: entries,
  }
  await writeFile(outPath, `${JSON.stringify(sortKeysDeep(output), null, 2)}\n`)
  console.log(`wrote ${entries.length} golden entries to ${outPath}`)
}

if (import.meta.main) {
  main().catch((error: unknown) => {
    console.error(String(error))
    process.exit(1)
  })
}
