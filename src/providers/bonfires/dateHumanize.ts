/**
 * Date-rendering pre-processor.
 *
 * Transforms ISO-8601 timestamps embedded in retrieved facts/edges/entity
 * summaries into human-readable forms BEFORE the answering LLM sees them.
 *
 * Background: gpt-4o-mini consistently fails to convert
 *   "...injury sustained last month. (2023-09-01T00:00:00+00:00)"
 * into "September 2023" — even with explicit prompt instructions and the
 * gold fact at rank #1. v70 hand-trace on q73 ("When did Melanie get
 * hurt?", gold "September 2023") confirmed: hypothesis was "October 13,
 * 2023" because the LLM grabbed the OTHER fact's event_time verbatim.
 *
 * Strategy: rewrite the strings server-side so the LLM never sees the
 * ISO form. We render a human date inline in two cases:
 *   1. Trailing parenthetical of the form `(YYYY-MM-DDT...)` or
 *      `(event_time: YYYY-MM-DD...)` — common on edges/episodes.
 *      Replace with `[occurred <DD Month YYYY>]`.
 *   2. Bare ISO timestamps embedded mid-text. Inline-replace with
 *      `<DD Month YYYY>`.
 *
 * Format choice: gold answers in LoCoMo predominantly use day-precision
 *   "13 August"            — q44
 *   "23 August 2023"       — q53
 *   "1 September 2023"     — q73 family
 *   "7 May 2023"           — many "When did X" questions
 * A 138-question survey of conv-26 alone shows `DD Month YYYY` is the
 * dominant gold form. So we ALWAYS emit day precision — even when the
 * source ISO is `T00:00:00` (graphiti's date-precision sentinel). The
 * day is in the source string in both cases (`2023-08-23` and
 * `2023-08-23T15:43:00`); surfacing it gives the answering LLM exact
 * day-level grounding for "When did X happen?" style questions.
 *
 * Returning the year always preserves disambiguation when conversations
 * span multiple years (this is rare in LoCoMo but matters elsewhere).
 *
 * Chunks ([CHUNK ...] / [PREFERENCE-SUMMARY ...] / [TOPIC-SUMMARY ...])
 * already carry a `[<tag> YYYY-MM-DD <speaker>]` prefix that the LLM
 * reads correctly — we leave those untouched.
 */

const MONTH_NAMES = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
]

interface ParsedIso {
  year: number
  month: number // 1-12
  day: number // 1-31
  hour: number
  minute: number
  second: number
  hasTime: boolean
}

/** Parse `YYYY-MM-DD[THH:MM[:SS][TZ]]` without using `Date` (avoids TZ
 * shifts on date-only strings). Returns null on bad input. */
export function parseIso(s: string): ParsedIso | null {
  const m = s.match(
    /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$/
  )
  if (!m) return null
  const year = parseInt(m[1], 10)
  const month = parseInt(m[2], 10)
  const day = parseInt(m[3], 10)
  if (month < 1 || month > 12 || day < 1 || day > 31) return null
  const hour = m[4] ? parseInt(m[4], 10) : 0
  const minute = m[5] ? parseInt(m[5], 10) : 0
  const second = m[6] ? parseInt(m[6], 10) : 0
  const hasTime = m[4] !== undefined && (hour !== 0 || minute !== 0 || second !== 0)
  return { year, month, day, hour, minute, second, hasTime }
}

/** Format a parsed ISO as a human-readable English date.
 *
 * Always emits day precision (`"23 August 2023"`) regardless of whether
 * the source ISO carried a real time-of-day. The day is in the source
 * string in both cases (`2023-08-23` and `2023-08-23T15:43:00`); surfacing
 * it gives the answering LLM exact day-level grounding for "When did X
 * happen?" questions whose LoCoMo gold answers (138/1986 conv-26
 * questions) overwhelmingly use the `DD Month YYYY` form. Time-of-day
 * is dropped — the embedder/LLM never need clock precision.
 *
 * `p.hasTime` is preserved on the struct for any consumer that cares
 * about the date-vs-datetime distinction, but the rendered string no
 * longer branches on it. */
export function formatHuman(p: ParsedIso): string {
  const monthName = MONTH_NAMES[p.month - 1]
  return `${p.day} ${monthName} ${p.year}`
}

/** Convenience: parse + format. Returns the original string if not a valid
 * ISO date (so callers can pipe through without conditional branching). */
export function humanizeIso(s: string): string {
  const p = parseIso(s)
  return p ? formatHuman(p) : s
}

// ---------------------------------------------------------------------------
// Text rewriting
//
// We handle three distinct shapes seen in the bench's rendered fact/edge text:
//
//   A) `(event_time: 2023-09-01T00:00:00+00:00)`  ← episodes, some facts
//   B) `(2023-09-01T00:00:00+00:00)`              ← graphiti edges
//   C) bare `2023-09-01T00:00:00+00:00` mid-text  ← rarer; fact summaries
//
// All three collapse to `[occurred <human>]` (parenthetical forms) or the
// inline `<human>` (bare form). We chose `[occurred ...]` over stripping
// entirely because the answer prompt's section instructions already lean
// on the LLM noticing event-time signals — leaving a tag preserves the
// "this is when it happened" intent, just in human form.
// ---------------------------------------------------------------------------

const TRAILING_EVENT_TIME_RE =
  /\s*\((?:event_time:\s*)?(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)\)\s*$/

const TRAILING_EVENT_TIME_ANY_RE = /\s*\(event_time:\s*([^)]+)\)\s*$/

const BARE_ISO_RE =
  /\b(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)\b/g

const MONTH_NAME_PATTERN =
  "January|February|March|April|May|June|July|August|September|October|November|December"

interface ParsedExplicitDate {
  key: string
  human: string
}

/** Rewrite a single rendered text string so embedded ISO dates become
 * human-readable. Idempotent — running on already-humanized text is a
 * no-op (no ISO patterns to match). */
export function humanizeDatesInText(text: string): string {
  if (typeof text !== "string" || text.length === 0) return text

  let out = text

  // Step 0: handle the generic HyperMem renderer shape first. If the fact
  // body itself states a full date and the appended event_time disagrees,
  // drop the appended timestamp; the body date is the event being recalled,
  // while the appended timestamp is often the message/session timestamp.
  const genericEventMatch = out.match(TRAILING_EVENT_TIME_ANY_RE)
  if (genericEventMatch) {
    const body = out.slice(0, genericEventMatch.index!)
    const bodyDate = parseExplicitDate(body)
    const eventDate = parseExplicitDate(genericEventMatch[1])
    if (bodyDate && eventDate && bodyDate.key !== eventDate.key) {
      out = body.trimEnd()
    } else if (eventDate) {
      out = `${body.trimEnd()} [occurred ${eventDate.human}]`
    }
  }

  // Step 1: handle trailing `(event_time: ...)` or `(YYYY-...)`
  // parentheticals. These are appended by search.ts's `${e.fact} (event_time: ${e.valid_at})`
  // template, so collapsing them to `[occurred <human>]` covers the most
  // common case cleanly without touching the fact body.
  const trailingMatch = out.match(TRAILING_EVENT_TIME_RE)
  if (trailingMatch) {
    const parsed = parseIso(trailingMatch[1])
    if (parsed) {
      out = out.slice(0, trailingMatch.index!) + ` [occurred ${formatHuman(parsed)}]`
    }
  }

  // Step 2: replace any remaining bare ISO strings inline. This catches
  // facts whose graphiti-extracted summary text mentions a date directly
  // (e.g., "...the 2023-09-15 hike was..."). We DO NOT touch dates that
  // appear inside square-bracket prefixes — chunk tags like
  //   `[CHUNK 2023-07-12 Caroline] ...`
  // already render dates the LLM reads correctly, and rewriting them
  // would break the `[date speaker]` triple convention.
  out = out.replace(BARE_ISO_RE, (match, iso, offset: number) => {
    // Skip dates inside bracketed prefixes: scan backwards from the match
    // position; if we hit `[` before `]` (or string start), we're inside
    // a tag and should leave the date alone.
    for (let i = offset - 1; i >= 0; i--) {
      const ch = out[i]
      if (ch === "]") break
      if (ch === "[") return match
    }
    const parsed = parseIso(iso)
    return parsed ? formatHuman(parsed) : match
  })

  return out
}

function parseExplicitDate(text: string): ParsedExplicitDate | null {
  const value = String(text || "")
  const iso = value.match(
    /\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b/
  )
  if (iso) {
    const parsed = parseIso(iso[0])
    if (parsed) return { key: `${iso[1]}-${iso[2]}-${iso[3]}`, human: formatHuman(parsed) }
  }

  const monthLookup = new Map(MONTH_NAMES.map((name, index) => [name.toLowerCase(), index + 1]))
  const monthFirst = value.match(
    new RegExp(`\\b(${MONTH_NAME_PATTERN})\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,\\s*|\\s+)(\\d{4})\\b`, "i")
  )
  if (monthFirst) {
    const month = monthLookup.get(monthFirst[1].toLowerCase())
    const day = parseInt(monthFirst[2], 10)
    const year = parseInt(monthFirst[3], 10)
    if (month && day >= 1 && day <= 31) {
      return {
        key: `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`,
        human: `${day} ${MONTH_NAMES[month - 1]} ${year}`,
      }
    }
  }

  const dayFirst = value.match(
    new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(${MONTH_NAME_PATTERN})\\s+(\\d{4})\\b`, "i")
  )
  if (dayFirst) {
    const day = parseInt(dayFirst[1], 10)
    const month = monthLookup.get(dayFirst[2].toLowerCase())
    const year = parseInt(dayFirst[3], 10)
    if (month && day >= 1 && day <= 31) {
      return {
        key: `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`,
        human: `${day} ${MONTH_NAMES[month - 1]} ${year}`,
      }
    }
  }
  return null
}

/** Apply the humanizer to every fact/entity/episode hit's `text` field.
 * Chunks pass through unchanged — their `[CHUNK YYYY-MM-DD speaker]`
 * tags are intentional date markers the LLM consumes correctly, and
 * the chunk body is conversational prose with already-human dates. */
export function humanizeHits<T extends { text: string; kind?: string }>(hits: T[]): T[] {
  return hits.map((h) => {
    if (h.kind === "chunk") return h
    return { ...h, text: humanizeDatesInText(h.text) }
  })
}
