import { minMax, monthWindow, type StatementEntry } from "./retrieval2"

export interface Comprehension {
  probe: string
  matched_cxn_ids: string[]
  bound_cxn_ids: string[]
  operators: { neg: boolean; modal: string | null }
  wh_slot: string | null
  fillers: Array<{ lemma: string; role: "subj" | "verb" | "obj"; entity: boolean }>
  date_fillers: Array<{ text: string; year: number | null; month: number | null }>
}

export function isFallback(c: Comprehension): boolean {
  return c.matched_cxn_ids.length === 0 && c.wh_slot === null && c.fillers.length === 0 && c.date_fillers.length === 0
}

// Affordance keys the comprehend sidecar can fire, per index.ts's search()
// (`affordancesFired.push(...)`) — kept as a literal tuple (not derived from
// the recipe) so gapmap.ts's computeAffordances can report all four keys at
// rate 0 for a run with zero v3 recipe metadata, rather than an empty object.
export const AFFORDANCE_KEYS = ["q:strata", "q:temporal", "q:seed", "q:answer"] as const
export type AffordanceKey = (typeof AFFORDANCE_KEYS)[number]

// corpus construct_ids (entry.construct_ids) are bound-firing-derived — the
// precision side, minted from what actually fired during the fold — while
// matchedIds (comprehension.matched_cxn_ids) are pattern-level — the recall
// side, whatever the sidecar's grammar matched against the query text. The
// superset (matchedIds) keys onto the subset's targets (construct_ids), and
// the effect is additive/boost-only, so this asymmetry can only over-fire a
// statement's score, never gate it out.
export function strataBoost(
  scores: Map<string, number>, statements: Map<string, StatementEntry>,
  matchedIds: string[], delta: number
): Map<string, number> {
  const out = new Map<string, number>()
  const matchedSet = new Set(matchedIds)
  for (const [hash, score] of scores) {
    if (matchedIds.length === 0) { out.set(hash, score); continue }
    const entry = statements.get(hash)
    const overlap = entry ? entry.construct_ids.filter((id) => id !== "residual.v1" && matchedSet.has(id)).length : 0
    out.set(hash, score + delta * overlap / matchedIds.length)
  }
  return out
}

export function temporalWindowFromDates(
  dateFillers: Comprehension["date_fillers"], defaultYear: number
): { fromTs: string; toTs: string } | null {
  // date_fillers[0] (alphabetical-by-text, kernel-sorted) is a deliberate
  // deterministic pick on multi-date questions — not "the first date
  // mentioned," just a stable choice.
  const first = dateFillers[0]
  if (!first) return null
  const year = first.year ?? defaultYear
  if (first.month !== null) return monthWindow(year, first.month - 1)
  return { fromTs: `${year}-01-01T00:00:00Z`, toTs: `${year}-12-31T23:59:59Z` }
}

function tokenize(lemma: string): string[] {
  return lemma.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 0)
}

export function seededTermWeights(
  fillers: Comprehension["fillers"], entityW: number, verbW: number
): Map<string, number> {
  const weights = new Map<string, number>()
  for (const filler of fillers) {
    const weight = filler.entity ? entityW : verbW
    for (const token of tokenize(filler.lemma)) {
      weights.set(token, Math.max(weights.get(token) ?? -Infinity, weight))
    }
  }
  return weights
}

export function combineSparse(
  natural: Map<string, number>, seeded: Map<string, number>, laneW: number
): Map<string, number> {
  const n = minMax(natural), s = minMax(seeded)
  const out = new Map<string, number>()
  for (const key of new Set([...n.keys(), ...s.keys()])) {
    out.set(key, (n.get(key) ?? 0) + laneW * (s.get(key) ?? 0))
  }
  return out
}

// Leg-4 v2 (spec 2026-07-08 §0.2): strings are part of the recipe — change
// them only with a spec amendment and a directiveVersion bump.
export function answerDirective(whSlot: string | null): string | null {
  switch (whSlot) {
    case "when":
      return "The question asks for a specific date or time. Resolve relative references in the evidence ('last Saturday', 'the week before') against that message's own timestamp, then answer with the most specific absolute date supported."
    case "list":
      return "The question asks for multiple items. Enumerate ALL distinct items supported by the evidence and the context window; do not stop at the first. Do not add items of a kind the question did not ask about."
    case "inference":
      return "This is an inference question. Reason from the evidence to a definite answer (e.g. yes/no or likely/unlikely) with a brief reason. Do NOT answer 'Not enough information' if the evidence supports a reasonable inference."
    case "how_long":
      return "Answer with the duration AND its absolute anchor (e.g. 'since 2016'), derived from the evidence timestamps if needed."
    case "how_many":
      return "Answer with a specific quantity or duration."
    case "who":
    case "where":
    case "what":
    case "which":
      return "Answer with the specific entity or fact, concisely."
    default:
      return null
  }
}

// Rides every slot directive; NEVER emitted alone (spec Decision 2).
export function directivePreamble(): string {
  return "State the concrete fact(s) in the evidence's own words rather than a vague paraphrase. Answer 'Not enough information' ONLY when the evidence and context contain nothing relevant to the question."
}
