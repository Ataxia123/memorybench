import { describe, expect, test } from "bun:test"
import { buildAnswerPromptV2, buildAnswerPromptV3 } from "../bonfires-cxn/retrieval2"
import { mapSearchEnvelope } from "./index"

describe("kernel evidence prompt parity", () => {
  test("absent optional evidence preserves the legacy prompt byte for byte", () => {
    const evidence = { text: "denial", kind: "edge", metadata: { source_text: "I did not go.", speaker: "Caroline" } }
    const carrier = { kind: "cxn_context", lines: ["context"], directive: "directive" }
    for (const render of [buildAnswerPromptV2, buildAnswerPromptV3]) {
      const original = render("Who?", [evidence, evidence, carrier], "2023-05-01")
      const mapped = render("Who?", [
        { ...evidence, temporal: null, answers: [] }, evidence, carrier,
      ], "2023-05-01")
      expect(mapped).toBe(original)
      expect(mapped.match(/I did not go\./g)).toHaveLength(1)
      expect(mapped).toContain("[Caroline] I did not go.")
    }
  })

  test("persisted envelope renders time provenance and recursively bound replies", () => {
    const items = mapSearchEnvelope({
      results: [{
        uuid: "parent", score: 1, text: "normalized claim", kind: "edge",
        metadata: { source_text: "I did not go.", speaker: "Caroline", polarity: "negative" },
        temporal: { iso: "2023-05-01", relative: "yesterday", source: "statement" },
        answers: [{
          uuid: "reply", score: 0.9, text: "Why?", metadata: { speaker: "Mara" },
          temporal: null,
          answers: [{ uuid: "nested", score: 0.8, text: "I was ill.", temporal: { source: "unknown" } }],
        }],
      }],
      context_lines: ["context"], directive: "directive", recipe: null, fallback: false,
    })
    const prompt = buildAnswerPromptV3("Who?", JSON.parse(JSON.stringify(items)))
    expect(prompt).toContain("[Caroline] I did not go. [temporal: iso=2023-05-01; relative=yesterday; source=statement]")
    expect(prompt).toContain('[reply to "I did not go."] Why?')
    expect(prompt).toContain('[reply to "Why?"] I was ill. [temporal: source=unknown]')
    expect(prompt).not.toContain("normalized claim")
    expect(prompt).not.toContain("undefined")
    expect(prompt).toContain("ANSWER DIRECTIVE:\ndirective")
    expect(buildAnswerPromptV2("Who?", items)).not.toContain("ANSWER DIRECTIVE:")
  })

  test("source deduplication does not discard a duplicate parent's unique reply", () => {
    const prompt = buildAnswerPromptV3("Who?", [
      { text: "first", metadata: { source_text: "Same source." } },
      { text: "second", metadata: { source_text: "Same source." }, answers: [{ text: "New reply." }] },
    ])
    expect(prompt).toContain('Same source.\n[reply to "Same source."] New reply.')
  })

  test("same source retains distinct temporal provenance but deduplicates identical evidence", () => {
    const first = { text: "claim", metadata: { source_text: "I was there." }, temporal: { iso: "2023-05-01", source: "statement" } }
    const second = { ...first, temporal: { iso: "2023-05-02", source: "statement" } }
    const prompt = buildAnswerPromptV3("When?", [first, second, first])
    expect(prompt).toContain("I was there. [temporal: iso=2023-05-01; source=statement]")
    expect(prompt).toContain("I was there. [temporal: iso=2023-05-02; source=statement]")
    expect(prompt.match(/I was there\./g)).toHaveLength(2)
  })

  test("a shared reply source retains both parent relationships", () => {
    const reply = { text: "normalized", metadata: { source_text: "Yes." } }
    const prompt = buildAnswerPromptV3("Who?", [
      { text: "Did Mara go?", answers: [reply] },
      { text: "Did Caroline go?", answers: [reply] },
    ])
    expect(prompt).toContain('[reply to "Did Mara go?"] Yes.')
    expect(prompt).toContain('[reply to "Did Caroline go?"] Yes.')
  })

  test("cyclic and excessively nested answer input fails explicitly", () => {
    const cyclic: Record<string, unknown> = { text: "cycle" }
    cyclic.answers = [cyclic]
    expect(() => buildAnswerPromptV3("Who?", [cyclic])).toThrow("Cyclic kernel answer evidence")
    let nested: Record<string, unknown> = { text: "leaf" }
    for (let depth = 0; depth < 66; depth++) nested = { text: "parent", answers: [nested] }
    expect(() => buildAnswerPromptV3("Who?", [nested])).toThrow("Kernel answer evidence exceeds nesting limit (64)")
  })
})
