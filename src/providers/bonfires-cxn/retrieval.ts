import type { UtteranceEntry } from "./artifacts"

// Static list — vendored, never fetched. Standard English function words plus
// question words; deliberately NO domain words.
export const STOPWORDS = new Set([
  "a","about","after","again","all","also","am","an","and","any","are","as","at",
  "be","because","been","before","being","between","both","but","by","can","could",
  "did","do","does","doing","down","during","each","few","for","from","further",
  "had","has","have","having","he","her","here","hers","him","his","how","i","if",
  "in","into","is","it","its","just","me","more","most","my","no","nor","not","now",
  "of","off","on","once","only","or","other","our","out","over","own","same","she",
  "should","so","some","such","than","that","the","their","them","then","there",
  "these","they","this","those","through","to","too","under","until","up","very",
  "was","we","were","what","when","where","which","while","who","whom","why","will",
  "with","would","you","your","yours","did","does","doing","get","got","go","going",
])

export function extractTerms(question: string): string[] {
  const tokens = question
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token))
  const bigrams: string[] = []
  for (let i = 0; i < tokens.length - 1; i++) bigrams.push(`${tokens[i]} ${tokens[i + 1]}`)
  const seen = new Set<string>()
  const out: string[] = []
  for (const term of [...tokens, ...bigrams]) {
    if (!seen.has(term)) {
      seen.add(term)
      out.push(term)
    }
  }
  return out
}

export interface CandidateFiring {
  uuid: string
  constructId: string
  utteranceHash: string
  ts: string
  matchedSeeds: Set<string>
}

export function rankFirings(
  candidates: CandidateFiring[],
  entrenchment: Map<string, number>,
  topK: number
): CandidateFiring[] {
  const score = (f: CandidateFiring) => entrenchment.get(f.constructId) ?? 0
  return [...candidates]
    .sort((a, b) => {
      if (b.matchedSeeds.size !== a.matchedSeeds.size) return b.matchedSeeds.size - a.matchedSeeds.size
      if (score(b) !== score(a)) return score(b) - score(a)
      if (a.ts !== b.ts) return a.ts < b.ts ? -1 : 1
      return a.uuid < b.uuid ? -1 : a.uuid > b.uuid ? 1 : 0
    })
    .slice(0, topK)
}

export interface StructureLine {
  firingUuid: string
  predicate: string
  roles: Array<{ role: string; filler: string }>
  ts: string
  constructId: string
}

export interface CxnSearchResult {
  text: string
  kind: "cxn_utterance"
  score: number
  metadata: { utterance_hash: string; firing_uuids: string[]; construct_ids: string[]; session: string }
}

function formatStamp(ts: string): string {
  // "2023-05-08T13:56:00Z" -> "2023-05-08 13:56"
  return ts.slice(0, 16).replace("T", " ")
}

export function assembleResults(
  ranked: CandidateFiring[],
  utteranceMap: Map<string, UtteranceEntry>,
  structures: StructureLine[]
): CxnSearchResult[] {
  const byHash = new Map<string, { entry: UtteranceEntry; firings: CandidateFiring[]; rankIndex: number }>()
  ranked.forEach((firing, index) => {
    const entry = utteranceMap.get(firing.utteranceHash)
    if (!entry) return // hydration miss: firing outside sidecar — preflight samples guard this; skip defensively
    const existing = byHash.get(firing.utteranceHash)
    if (existing) existing.firings.push(firing)
    else byHash.set(firing.utteranceHash, { entry, firings: [firing], rankIndex: index })
  })

  const utteranceResults: CxnSearchResult[] = [...byHash.entries()]
    .sort((a, b) => (a[1].entry.ts < b[1].entry.ts ? -1 : a[1].entry.ts > b[1].entry.ts ? 1 : a[0] < b[0] ? -1 : 1))
    .map(([hash, { entry, firings, rankIndex }]) => ({
      text: `[${formatStamp(entry.ts)} ${entry.actor_id}] ${entry.utterance}`,
      kind: "cxn_utterance" as const,
      score: 1 / (1 + rankIndex),
      metadata: {
        utterance_hash: hash,
        firing_uuids: firings.map((f) => f.uuid).sort(),
        construct_ids: [...new Set(firings.map((f) => f.constructId))].sort(),
        session: entry.session,
      },
    }))

  return utteranceResults
}

export function buildStructureItem(structures: StructureLine[]): { kind: "cxn_structure"; lines: string[] } {
  return {
    kind: "cxn_structure",
    lines: structures
      .map(
        (s) =>
          `${s.predicate}(${s.roles.map((r) => `${r.role}=${r.filler}`).join(", ")}) @ ${s.ts} [${s.constructId}]`
      )
      .sort(),
  }
}

export function buildCxnAnswerPrompt(question: string, context: unknown[], questionDate?: string): string {
  const utterances: string[] = []
  const structureLines: string[] = []
  for (const item of context) {
    if (!item || typeof item !== "object") continue
    const record = item as Record<string, unknown>
    if (record.kind === "cxn_utterance" && typeof record.text === "string") utterances.push(record.text)
    if (record.kind === "cxn_structure" && Array.isArray(record.lines)) {
      for (const line of record.lines) structureLines.push(String(line))
    }
  }
  const dateLine = questionDate ? `\nThe question is asked on: ${questionDate}` : ""
  return `You are answering a question about a two-person conversation using retrieved evidence.

EVIDENCE (conversation excerpts, chronological — each line is "[date time speaker] utterance"):
${utterances.join("\n")}

STRUCTURE (events extracted from the conversation as predicate(role=filler) @ timestamp):
${structureLines.join("\n")}
${dateLine}

Question: ${question}

Answer concisely using ONLY the evidence above. If the evidence does not contain the answer, say "Not enough information."`
}
