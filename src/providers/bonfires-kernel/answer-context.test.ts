import { describe, expect, test } from "bun:test"
import { buildDefaultAnswerPrompt } from "../../prompts/defaults"
import { buildContextString } from "../../types/prompts"
import { buildAnswerPromptV3 } from "../bonfires-cxn/retrieval2"
import { BonfiresKernelProvider, mapSearchEnvelope } from "./index"
import { buildKernelAnswerPrompt } from "./prompts"

describe("actual kernel answer hook context boundary", () => {
  test("non-evidence provenance growth does not grow the actual answer prompt", () => {
    const render = new BonfiresKernelProvider().prompts.answerPrompt
    expect(render).toBe(buildKernelAnswerPrompt)
    const evidence = { text: "I stayed home.", kind: "edge", score: 1, metadata: { speaker: "Mara" } }
    const carrier = { kind: "cxn_context", lines: ["Mara: I stayed home."], directive: "Be concise." }
    const lean = [evidence, carrier]
    const inflated = [
      { ...evidence, uuid: "x".repeat(100000), family: ["y".repeat(100000)],
        episode_ids: ["z".repeat(100000)], answers: [], temporal: null },
      { ...carrier, recipe: { diagnostics: "d".repeat(100000) }, kind_mix: { edge: 1 }, degraded: {} },
    ]
    expect(JSON.stringify(inflated).length).toBeGreaterThan(JSON.stringify(lean).length + 399999)
    expect(render("Where?", inflated)).toBe(render("Where?", lean))
  })

  test("answer rendering leaves persisted wire provenance byte-identical", () => {
    const items = mapSearchEnvelope({
      results: [{
        uuid: "parent", text: "Where?", kind: "edge", score: 1, metadata: {},
        family: ["construction"], episode_ids: ["episode"],
        answers: [{ uuid: "child", text: "Home.", score: 0.8, temporal: { iso: "2023-05-01" } }],
      }],
      context_lines: ["raw turn"], directive: "Use evidence.", recipe: { diagnostic: "retain" }, fallback: false,
    })
    const bytes = JSON.stringify(items)
    const persisted = JSON.parse(bytes)
    new BonfiresKernelProvider().prompts.answerPrompt("Where?", persisted)
    expect(JSON.stringify(persisted)).toBe(bytes)
    expect(persisted[0].family).toEqual(["construction"])
    expect(persisted[0].episode_ids).toEqual(["episode"])
    expect(persisted[0].answers[0].uuid).toBe("child")
    expect(persisted[1].recipe.diagnostic).toBe("retain")
  })

  test("nested replies and temporal evidence are rendered, not discarded as diagnostics", () => {
    const render = new BonfiresKernelProvider().prompts.answerPrompt
    const context = [{
      text: "Where were you?",
      answers: [{
        text: "normalized", metadata: { source_text: "Home.", speaker: "Mara", valid_at: "2023-05-01T12:00:00Z" },
        temporal: { iso: "2023-05-01", relative: "yesterday", source: "statement", diagnostic: "omit" },
        answers: [{ text: "Why?", temporal: { first_fired: "2023-05-01", last_fired: "2023-05-02" } }],
      }],
    }, { kind: "cxn_context", lines: ["raw conversation"], directive: "Use dates." }]
    const prompt = render("Where?", context, "2023-05-02")
    expect(prompt).toContain('[reply to "Where were you?"] [2023-05-01 12:00 Mara] Home.')
    expect(prompt).toContain('[reply to "Home."] Why?')
    expect(prompt).toContain("first_fired=2023-05-01; last_fired=2023-05-02")
    expect(prompt).toContain("ANSWER DIRECTIVE:\nUse dates.")
    expect(prompt).not.toContain("diagnostic")
  })

  test("ordinary default-provider JSON formatting is unchanged", () => {
    const context = [{ text: "Evidence", custom_provider_field: "keep me", metadata: { custom: 7 } }]
    const serialized = JSON.stringify(context, null, 2)
    expect(buildContextString(context)).toBe(serialized)
    expect(buildDefaultAnswerPrompt("Question?", context)).toContain(serialized)
  })

  test("fallback dates and clock-relative text are omitted recursively without changing stored hits", () => {
    const temporal = { iso: "2023-05-08", relative: "3 years ago", source: "statement_created_at" }
    const plain = [{
      text: "[2023-05-08 Caroline] I went on May 7.",
      metadata: { temporal },
      answers: [{ text: "How was it?", answers: [{ text: "Powerful." }] }],
    }]
    const fallback = [{
      ...plain[0], temporal,
      answers: [{ text: "How was it?", temporal, answers: [{ text: "Powerful.", temporal }] }],
    }]
    const bytes = JSON.stringify(fallback)
    const render = new BonfiresKernelProvider().prompts.answerPrompt
    expect(render("When?", fallback)).toBe(render("When?", plain))
    expect(render("When?", fallback)).not.toContain("3 years ago")
    expect(render("When?", fallback)).not.toContain("statement_created_at")
    expect(render("When?", fallback)).toContain("[2023-05-08 Caroline] I went on May 7.")
    expect(JSON.stringify(fallback)).toBe(bytes)
    // The shared CXN provider renderer keeps its pre-existing temporal behavior.
    expect(buildAnswerPromptV3("When?", fallback)).toContain("statement_created_at")
  })

  test("non-fallback event dates and firing windows retain byte-identical rendering", () => {
    const context = [
      { text: "The event.", temporal: { source: "edge_valid_at", iso: "2023-05-07" } },
      { text: "A firing.", temporal: { source: "firing_window", first_fired: "2023-05-01", last_fired: "2023-05-02" } },
    ]
    expect(buildKernelAnswerPrompt("When?", context)).toBe(buildAnswerPromptV3("When?", context))
  })

  test("cyclic and excessive reply nesting still fail explicitly", () => {
    const cyclic: Record<string, unknown> = { text: "cycle" }
    cyclic.answers = [cyclic]
    expect(() => buildKernelAnswerPrompt("Who?", [cyclic])).toThrow("Cyclic kernel answer evidence")
    let nested: Record<string, unknown> = { text: "leaf" }
    for (let depth = 0; depth < 66; depth++) nested = { text: "parent", answers: [nested] }
    expect(() => buildKernelAnswerPrompt("Who?", [nested])).toThrow("Kernel answer evidence exceeds nesting limit (64)")
  })

  test("distinct dated occurrences survive hidden fallback annotations while exact duplicates deduplicate", () => {
    const may = {
      text: "normalized", metadata: { source_text: "I went swimming.", valid_at: "2023-05-01T12:00:00Z" },
      temporal: { iso: "2023-05-01", relative: "3 years ago", source: "statement_created_at" },
    }
    const june = {
      ...may, metadata: { ...may.metadata, valid_at: "2023-06-01T12:00:00Z" },
      temporal: { ...may.temporal, iso: "2023-06-01" },
    }
    const context = [may, june, may]
    const bytes = JSON.stringify(context)
    const prompt = buildKernelAnswerPrompt("When?", context)
    expect(prompt).toContain("[2023-05-01 12:00] I went swimming.")
    expect(prompt).toContain("[2023-06-01 12:00] I went swimming.")
    expect(prompt.match(/I went swimming\./g)).toHaveLength(2)
    expect(prompt).not.toContain("[temporal:")
    expect(JSON.stringify(context)).toBe(bytes)
    expect(buildAnswerPromptV3("When?", context).match(/I went swimming\./g)).toHaveLength(2)
    expect(buildAnswerPromptV3("When?", context)).toContain("statement_created_at")

    const replies = [{ text: "What did you do?", answers: [may, june, may] }]
    const replyBytes = JSON.stringify(replies)
    const nestedPrompt = buildKernelAnswerPrompt("When?", replies)
    expect(nestedPrompt).toContain('[reply to "What did you do?"] [2023-05-01 12:00] I went swimming.')
    expect(nestedPrompt).toContain('[reply to "What did you do?"] [2023-06-01 12:00] I went swimming.')
    expect(nestedPrompt.match(/I went swimming\./g)).toHaveLength(2)
    expect(nestedPrompt).not.toContain("[temporal:")
    expect(JSON.stringify(replies)).toBe(replyBytes)
  })
})
