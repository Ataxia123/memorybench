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

export function answerDirective(whSlot: string | null): string | null {
  switch (whSlot) {
    case "when":
      return "The question asks for a specific date or time. Answer with the most specific date or time supported by the evidence; derive it from the evidence timestamps if needed."
    case "list":
      return "The question asks for multiple items. Enumerate ALL items supported by the evidence."
    case "how_many":
    case "how_long":
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
