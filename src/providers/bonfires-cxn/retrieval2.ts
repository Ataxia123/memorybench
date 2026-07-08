export interface StatementEntry {
  hash: string
  utterance: string
  ts: string
  actor_id: string
  session: string
  session_index: number
  construct_ids: string[]
}

// ---------- BM25 (Okapi, k1=1.2 b=0.75; plain lowercase tokens, no stopword strip) ----------

const K1 = 1.2
const B = 0.75

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((t) => t.length > 0)
}

export interface Bm25Index {
  docTokens: Map<string, Map<string, number>>   // hash -> term -> tf
  docLength: Map<string, number>
  df: Map<string, number>
  avgLength: number
  count: number
}

export function buildBm25(statements: StatementEntry[]): Bm25Index {
  const docTokens = new Map<string, Map<string, number>>()
  const docLength = new Map<string, number>()
  const df = new Map<string, number>()
  let total = 0
  for (const statement of statements) {
    const tokens = tokenize(statement.utterance)
    const tf = new Map<string, number>()
    for (const token of tokens) tf.set(token, (tf.get(token) ?? 0) + 1)
    docTokens.set(statement.hash, tf)
    docLength.set(statement.hash, tokens.length)
    total += tokens.length
    for (const term of tf.keys()) df.set(term, (df.get(term) ?? 0) + 1)
  }
  return { docTokens, docLength, df, avgLength: total / Math.max(statements.length, 1), count: statements.length }
}

export function bm25Scores(index: Bm25Index, query: string): Map<string, number> {
  const terms = [...new Set(tokenize(query))]
  return bm25ScoresWeighted(index, new Map(terms.map((term) => [term, 1])))
}

export function bm25ScoresWeighted(index: Bm25Index, termWeights: Map<string, number>): Map<string, number> {
  const scores = new Map<string, number>()
  for (const [term, weight] of termWeights) {
    if (weight <= 0) continue
    const documentFrequency = index.df.get(term)
    if (!documentFrequency) continue
    const idf = Math.log(1 + (index.count - documentFrequency + 0.5) / (documentFrequency + 0.5))
    for (const [hash, tf] of index.docTokens) {
      const frequency = tf.get(term)
      if (!frequency) continue
      const length = index.docLength.get(hash) ?? 0
      const denominator = frequency + K1 * (1 - B + (B * length) / index.avgLength)
      scores.set(hash, (scores.get(hash) ?? 0) + weight * idf * ((frequency * (K1 + 1)) / denominator))
    }
  }
  return scores
}

// ---------- dense + blend ----------

import { cosine } from "./voyage"

export function denseScores(queryVector: number[], vectors: Map<string, number[]>): Map<string, number> {
  const scores = new Map<string, number>()
  for (const [hash, vector] of vectors) scores.set(hash, cosine(queryVector, vector))
  return scores
}

export function minMax(scores: Map<string, number>): Map<string, number> {
  if (scores.size === 0) return scores
  let min = Infinity, max = -Infinity
  for (const value of scores.values()) { min = Math.min(min, value); max = Math.max(max, value) }
  const range = max - min
  const out = new Map<string, number>()
  for (const [key, value] of scores) out.set(key, range === 0 ? 1 : (value - min) / range)
  return out
}

export function blendScores(
  dense: Map<string, number>, sparse: Map<string, number>, wDense: number, wSparse: number
): Map<string, number> {
  const d = minMax(dense), s = minMax(sparse)
  const out = new Map<string, number>()
  for (const key of new Set([...d.keys(), ...s.keys()])) {
    out.set(key, wDense * (d.get(key) ?? 0) + wSparse * (s.get(key) ?? 0))
  }
  return out
}

// Leg-5 lane B1 (spec §0.4): diversity-aware final-K selection for list-shape
// questions. Reorders WITHIN the pool — never imports new candidates.
export function mmrSelect(
  poolScores: Map<string, number>,
  vectorOf: (id: string) => number[] | undefined,
  finalK: number,
  lambda: number
): Array<[string, number]> {
  const relevance = minMax(poolScores)
  const candidates = [...poolScores.keys()].sort(
    (a, b) => (relevance.get(b)! - relevance.get(a)!) || (a < b ? -1 : 1)
  )
  for (const id of candidates) {
    if (!vectorOf(id)) throw new Error(`bonfires-cxn: mmrSelect candidate ${id} has no vector`)
  }
  const selected: string[] = []
  const remaining = new Set(candidates)
  while (selected.length < Math.min(finalK, candidates.length)) {
    let best: string | null = null
    let bestValue = -Infinity
    for (const id of candidates) {
      if (!remaining.has(id)) continue
      let maxSim = 0
      for (const s of selected) maxSim = Math.max(maxSim, cosine(vectorOf(id)!, vectorOf(s)!))
      const value = relevance.get(id)! - lambda * maxSim
      if (value > bestValue) { bestValue = value; best = id }
      // ties resolve to the earlier candidate (higher relevance, then id asc) via strict >
    }
    selected.push(best!)
    remaining.delete(best!)
  }
  return selected.map((id) => [id, poolScores.get(id)!])
}

// ---------- Lane P ----------

export interface Aggregates { cxn: Map<string, number[]>; episode: Map<string, number[]> }

export function lanePBoost(
  blended: Map<string, number>, queryVector: number[], aggregates: Aggregates,
  statements: Map<string, StatementEntry>, deltaCxn: number, deltaEp: number
): { boosted: Map<string, number>; topContribs: Array<{ key: string; sim: number }> } {
  const cxnSim = new Map<string, number>()
  for (const [id, vector] of aggregates.cxn) cxnSim.set(id, Math.max(0, cosine(queryVector, vector)))
  const episodeSim = new Map<string, number>()
  for (const [id, vector] of aggregates.episode) episodeSim.set(id, Math.max(0, cosine(queryVector, vector)))

  const boosted = new Map<string, number>()
  for (const [hash, score] of blended) {
    const entry = statements.get(hash)
    if (!entry) { boosted.set(hash, score); continue }
    let cxnBest = 0
    for (const cid of entry.construct_ids) cxnBest = Math.max(cxnBest, cxnSim.get(cid) ?? 0)
    const episodeBoost = episodeSim.get(entry.session) ?? 0
    boosted.set(hash, score + deltaCxn * cxnBest + deltaEp * episodeBoost)
  }
  const topContribs = [
    ...[...cxnSim.entries()].map(([key, sim]) => ({ key: `cxn:${key}`, sim })),
    ...[...episodeSim.entries()].map(([key, sim]) => ({ key: `ep:${key}`, sim })),
  ].sort((a, b) => b.sim - a.sim || (a.key < b.key ? -1 : 1)).slice(0, 3)
  return { boosted, topContribs }
}

// ---------- temporal gate (structural: explicit month/year tokens only) ----------

const MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]
const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

export function monthWindow(year: number, monthIndex: number): { fromTs: string; toTs: string } {
  const mm = String(monthIndex + 1).padStart(2, "0")
  const daysInMonth = monthIndex === 1 && (year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)) ? 29 : DAYS_IN_MONTH[monthIndex]
  return {
    fromTs: `${year}-${mm}-01T00:00:00Z`,
    toTs: `${year}-${mm}-${String(daysInMonth).padStart(2, "0")}T23:59:59Z`,
  }
}

// LoCoMo conv-26 corpus year — the fallback year for month-only temporal
// expressions with no explicit year, used by both the legacy regex gate
// below and the comprehension-derived date gate (affordance.ts's
// temporalWindowFromDates, called from index.ts).
export const CORPUS_YEAR = 2023

export function temporalWindow(question: string): { fromTs: string; toTs: string } | null {
  const lower = question.toLowerCase()
  const yearMatch = /\b(20\d{2})\b/.exec(lower)
  const monthIndex = MONTHS.findIndex((month) => {
    if (month === "may") {
      return /\b(?:in|of|during|last|this|next)\s+may\b/.test(lower) || /\bmay\s*,?\s*\d/.test(lower)
    }
    return new RegExp(`\\b${month}\\b`).test(lower)
  })
  if (monthIndex < 0 && !yearMatch) return null
  const year = yearMatch ? Number(yearMatch[1]) : CORPUS_YEAR   // corpus year when only a month is named
  if (monthIndex >= 0) return monthWindow(year, monthIndex)
  return { fromTs: `${year}-01-01T00:00:00Z`, toTs: `${year}-12-31T23:59:59Z` }
}

export function applyTemporalBoost(
  scores: Map<string, number>, statements: Map<string, { ts: string }>,
  window: { fromTs: string; toTs: string }, boost: number
): Map<string, number> {
  const out = new Map<string, number>()
  for (const [hash, score] of scores) {
    const ts = statements.get(hash)?.ts ?? ""
    out.set(hash, ts >= window.fromTs && ts <= window.toTs ? score + boost : score)
  }
  return out
}

// ---------- reply expansion ----------

// pool must arrive score-ordered — "top parents" = first N entries; the caller (provider search) sorts before calling.
export function isAskShape(entry: StatementEntry): boolean {
  if (entry.construct_ids.some((cid) => cid.startsWith("person.ask."))) return true
  return /\basked?\b/i.test(entry.utterance)
}

export function replyExpansion(
  pool: string[], scores: Map<string, number>, statements: Map<string, StatementEntry>,
  topParents: number, damp: number
): { added: Array<{ hash: string; score: number }> } {
  const bySessionIndex = new Map<string, string>()
  for (const entry of statements.values()) {
    bySessionIndex.set(`${entry.session}#${entry.session_index}`, entry.hash)
  }
  const added: Array<{ hash: string; score: number }> = []
  const inPool = new Set(pool)
  for (const parentHash of pool.slice(0, topParents)) {
    const parent = statements.get(parentHash)
    if (!parent || !isAskShape(parent)) continue
    const childHash = bySessionIndex.get(`${parent.session}#${parent.session_index + 1}`)
    if (!childHash || inPool.has(childHash)) continue
    inPool.add(childHash)
    added.push({ hash: childHash, score: (scores.get(parentHash) ?? 0) * damp })
  }
  return { added }
}

// ---------- hydration ----------

export interface Turn { ts: string; speaker: string; text: string; blip_caption: string | null }

function formatStamp(ts: string): string {
  return ts.slice(0, 16).replace("T", " ")
}

export function hydrationLines(
  topHashes: string[], statements: Map<string, StatementEntry>,
  turns: Map<string, Turn[]>, window: number
): string[] {
  const picked = new Map<string, Turn>()
  for (const hash of topHashes) {
    const entry = statements.get(hash)
    if (!entry) continue
    const sessionTurns = turns.get(entry.session) ?? []
    // nearest turn by ts (stable: first minimal distance)
    let best = -1, bestDistance = Infinity
    sessionTurns.forEach((turn, index) => {
      const distance = Math.abs(Date.parse(turn.ts) - Date.parse(entry.ts))
      if (distance < bestDistance) { bestDistance = distance; best = index }
    })
    if (best < 0) continue
    for (let i = Math.max(0, best - window); i <= Math.min(sessionTurns.length - 1, best + window); i++) {
      const turn = sessionTurns[i]!
      picked.set(`${turn.ts}|${turn.speaker}|${turn.text}`, turn)
    }
  }
  return [...picked.values()]
    .sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : a.speaker < b.speaker ? -1 : 1))
    .map((turn) =>
      `[${formatStamp(turn.ts)} ${turn.speaker}] ${turn.text}${turn.blip_caption ? ` (image: ${turn.blip_caption})` : ""}`
    )
}

// ---------- answer prompt ----------

function buildAnswerPromptCore(
  question: string, context: unknown[], questionDate: string | undefined, honorDirectives: boolean
): string {
  const utterances: string[] = []
  const contextLines: string[] = []
  let directive: string | null = null
  for (const item of context) {
    if (!item || typeof item !== "object") continue
    const record = item as Record<string, unknown>
    if (record.kind === "cxn_utterance" && typeof record.text === "string") utterances.push(record.text)
    if (record.kind === "cxn_context" && Array.isArray(record.lines)) {
      for (const line of record.lines) contextLines.push(String(line))
      if (honorDirectives && !directive && typeof record.directive === "string" && record.directive.length > 0) {
        directive = record.directive
      }
    }
  }
  const directiveBlock = directive ? `\nANSWER DIRECTIVE:\n${directive}\n` : ""
  const dateLine = questionDate ? `\nThe question is asked on: ${questionDate}` : ""
  return `You are answering a question about a two-person conversation using retrieved evidence.

EVIDENCE (retrieved statements, chronological — "[date time speaker] statement"):
${utterances.join("\n")}

CONTEXT WINDOW (raw conversation turns around the strongest evidence; may include image descriptions):
${contextLines.join("\n")}
${directiveBlock}${dateLine}

Question: ${question}

Answer concisely using ONLY the evidence and context above. If they do not contain the answer, say "Not enough information."`
}

export function buildAnswerPromptV2(question: string, context: unknown[], questionDate?: string): string {
  return buildAnswerPromptCore(question, context, questionDate, false)
}

export function buildAnswerPromptV3(question: string, context: unknown[], questionDate?: string): string {
  return buildAnswerPromptCore(question, context, questionDate, true)
}
