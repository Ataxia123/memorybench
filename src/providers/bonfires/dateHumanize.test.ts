import { describe, it, expect } from "bun:test"
import {
  parseIso,
  formatHuman,
  humanizeIso,
  humanizeDatesInText,
  humanizeHits,
} from "./dateHumanize.js"

describe("parseIso", () => {
  it("parses date-only YYYY-MM-DD", () => {
    expect(parseIso("2023-09-01")).toEqual({
      year: 2023,
      month: 9,
      day: 1,
      hour: 0,
      minute: 0,
      second: 0,
      hasTime: false,
    })
  })

  it("parses ISO with zero-time as date-only (hasTime=false)", () => {
    // The graphiti-emitted form for date-precision facts. We treat
    // 00:00:00 as a synthetic time component (no real clock-time
    // intent). The hasTime flag is preserved on the struct, but the
    // rendered output (formatHuman) no longer branches on it — both
    // forms render with day precision (`<DD Month YYYY>`).
    const p = parseIso("2023-09-01T00:00:00+00:00")
    expect(p?.hasTime).toBe(false)
    expect(p?.year).toBe(2023)
    expect(p?.month).toBe(9)
    expect(p?.day).toBe(1)
  })

  it("parses ISO with non-zero time as datetime (hasTime=true)", () => {
    const p = parseIso("2023-08-14T15:43:00+00:00")
    expect(p?.hasTime).toBe(true)
    expect(p?.year).toBe(2023)
    expect(p?.month).toBe(8)
    expect(p?.day).toBe(14)
    expect(p?.hour).toBe(15)
    expect(p?.minute).toBe(43)
  })

  it("parses Z-suffix UTC", () => {
    const p = parseIso("2023-03-15T16:33:00Z")
    expect(p?.hasTime).toBe(true)
    expect(p?.month).toBe(3)
  })

  it("parses fractional seconds", () => {
    const p = parseIso("2023-08-14T15:43:00.123+00:00")
    expect(p?.hasTime).toBe(true)
  })

  it("rejects garbage", () => {
    expect(parseIso("not-a-date")).toBe(null)
    expect(parseIso("")).toBe(null)
    expect(parseIso("2023-13-01")).toBe(null) // invalid month
    expect(parseIso("2023-09-32")).toBe(null) // invalid day
  })
})

describe("formatHuman", () => {
  it("formats date-only as '<DD> <Month> <YYYY>'", () => {
    const p = parseIso("2023-09-01")
    expect(formatHuman(p!)).toBe("1 September 2023")
  })

  it("formats datetime as '<DD> <Month> <YYYY>'", () => {
    const p = parseIso("2023-08-14T15:43:00+00:00")
    expect(formatHuman(p!)).toBe("14 August 2023")
  })

  it("zero-time ISO keeps day precision", () => {
    // q73 case: "2023-09-01T00:00:00+00:00" → "1 September 2023"
    // (was "September 2023" pre day-precision upgrade — gold answers
    // in LoCoMo use day-precision form, so always include day.)
    const p = parseIso("2023-09-01T00:00:00+00:00")
    expect(formatHuman(p!)).toBe("1 September 2023")
  })
})

describe("humanizeIso", () => {
  it("round-trips correctly for the q73 gold-fact ISO (day-precision)", () => {
    expect(humanizeIso("2023-09-01T00:00:00+00:00")).toBe("1 September 2023")
  })

  it("handles 14 August (q44 daughter's birthday)", () => {
    expect(humanizeIso("2023-08-14T15:43:00+00:00")).toBe("14 August 2023")
  })

  it("returns input unchanged on invalid ISO", () => {
    expect(humanizeIso("hello world")).toBe("hello world")
  })
})

describe("humanizeDatesInText", () => {
  it("rewrites trailing (event_time: ISO) parenthetical", () => {
    const input =
      "Melanie had to stop her pottery practice due to an injury sustained last month. (event_time: 2023-09-01T00:00:00+00:00)"
    expect(humanizeDatesInText(input)).toBe(
      "Melanie had to stop her pottery practice due to an injury sustained last month. [occurred 1 September 2023]"
    )
  })

  it("drops conflicting event_time when the fact body already states a full date", () => {
    const input =
      "Caroline attended an LGBTQ support group on May 7, 2023. (event_time: 2023-05-08T14:04:00.000Z)"
    expect(humanizeDatesInText(input)).toBe("Caroline attended an LGBTQ support group on May 7, 2023.")
  })

  it("renders human event_time through the same occurred format", () => {
    const input = "Caroline attended a support group. (event_time: 7 May 2023)"
    expect(humanizeDatesInText(input)).toBe("Caroline attended a support group. [occurred 7 May 2023]")
  })

  it("rewrites trailing bare (ISO) parenthetical (graphiti edge form)", () => {
    // This is the exact shape the v70 fact at q73 rank #1 uses.
    const input =
      "Melanie had to stop her pottery practice due to an injury sustained last month. (2023-09-01T00:00:00+00:00)"
    expect(humanizeDatesInText(input)).toBe(
      "Melanie had to stop her pottery practice due to an injury sustained last month. [occurred 1 September 2023]"
    )
  })

  it("rewrites trailing parenthetical with non-zero time as full date", () => {
    const input = "Caroline went to the vet. (2023-08-23T15:43:00+00:00)"
    expect(humanizeDatesInText(input)).toBe("Caroline went to the vet. [occurred 23 August 2023]")
  })

  it("[FACT]-prefixed text retains the prefix and humanizes the trailing date", () => {
    const input = "[FACT] Melanie practiced pottery for seven years. (2023-09-01T00:00:00+00:00)"
    expect(humanizeDatesInText(input)).toBe(
      "[FACT] Melanie practiced pottery for seven years. [occurred 1 September 2023]"
    )
  })

  it("rewrites bare ISO mid-text", () => {
    const input = "The hike on 2023-08-23T00:00:00+00:00 was great."
    expect(humanizeDatesInText(input)).toBe("The hike on 23 August 2023 was great.")
  })

  it("rewrites bare YYYY-MM-DD mid-text", () => {
    const input = "Melanie practiced on 2023-09-15 with her family."
    expect(humanizeDatesInText(input)).toBe("Melanie practiced on 15 September 2023 with her family.")
  })

  it("does NOT touch dates inside [CHUNK ...] tags", () => {
    // Chunk tag dates are conventional markers the LLM reads correctly.
    // Rewriting would break the [date speaker] triple convention.
    const input = "[CHUNK 2023-07-12 Caroline] Glad it helped ya, Melanie!"
    expect(humanizeDatesInText(input)).toBe(input)
  })

  it("does NOT touch dates inside [PREFERENCE-SUMMARY ...] tags", () => {
    const input = "[PREFERENCE-SUMMARY 2023-07-20 Caroline] Likes hiking and pottery."
    expect(humanizeDatesInText(input)).toBe(input)
  })

  it("is a no-op when no ISO date is present", () => {
    const input = "Melanie and Caroline provide mutual support for one another."
    expect(humanizeDatesInText(input)).toBe(input)
  })

  it("is idempotent — rerunning yields the same output", () => {
    const input = "Melanie hurt herself. (2023-09-01T00:00:00+00:00)"
    const once = humanizeDatesInText(input)
    const twice = humanizeDatesInText(once)
    expect(twice).toBe(once)
  })

  it("handles empty / undefined-ish input safely", () => {
    expect(humanizeDatesInText("")).toBe("")
  })

  it("rewrites multiple bare ISO dates in one string", () => {
    const input = "From 2023-06-01 to 2023-09-01 she practiced."
    expect(humanizeDatesInText(input)).toBe("From 1 June 2023 to 1 September 2023 she practiced.")
  })
})

describe("humanizeHits", () => {
  it("rewrites fact and entity hits, leaves chunks alone", () => {
    const hits = [
      {
        text: "Melanie hurt herself. (2023-09-01T00:00:00+00:00)",
        kind: "fact" as const,
      },
      {
        text: "[CHUNK 2023-07-12 Caroline] Glad it helped",
        kind: "chunk" as const,
      },
      {
        text: "melanie: She has been hiking since 2023-08-23 with her family.",
        kind: "entity" as const,
      },
    ]
    const out = humanizeHits(hits)
    expect(out[0].text).toBe("Melanie hurt herself. [occurred 1 September 2023]")
    expect(out[1].text).toBe("[CHUNK 2023-07-12 Caroline] Glad it helped") // unchanged
    expect(out[2].text).toBe(
      "melanie: She has been hiking since 23 August 2023 with her family."
    )
  })

  it("preserves all non-text fields unchanged", () => {
    const hits = [
      {
        text: "fact body. (2023-09-01T00:00:00+00:00)",
        score: 0.42,
        kind: "fact" as const,
      },
    ]
    const out = humanizeHits(hits)
    expect(out[0].score).toBe(0.42)
    expect(out[0].kind).toBe("fact")
  })

  it("handles hits without an explicit kind (treated as non-chunk → humanized)", () => {
    const hits = [{ text: "alpha (2023-09-01T00:00:00+00:00)" }]
    const out = humanizeHits(hits)
    expect(out[0].text).toBe("alpha [occurred 1 September 2023]")
  })
})
