import { expect, test } from "bun:test"
import { mapShimReply, type ShimReply } from "./index"
import { buildKernelAnswerPrompt } from "../bonfires-kernel/prompts"
import { createProvider, getAvailableProviders } from "../index"

const reply: ShimReply = {
  context:
    "## Evidence\n[S0 · 2023-05-08 · Caroline → Melanie] I went to a group.\n\n## Windows\nline",
  message_ids: ["D1:1"],
  tokens: 12,
  directive: "Answer carefully.",
  frame: { shape: "when", label: null },
  hits: [{ rank: 1, message_id: "D1:1", text: "hit text", score: 0.5 }],
  recipe: { key_recipe_tag: "legacy|deixis=on|v4", group: "g1-conv26-round2" },
}

test("registered under its name", () => {
  expect(getAvailableProviders()).toContain("bonfires-graph-ctx")
  expect(createProvider("bonfires-graph-ctx").name).toBe("bonfires-graph-ctx")
})

test("evidence section -> EVIDENCE, other sections verbatim -> CONTEXT WINDOW, directive, no hit text", () => {
  const items = mapShimReply(reply)
  const prompt = buildKernelAnswerPrompt("When?", items)
  const ev = "[S0 · 2023-05-08 · Caroline → Melanie] I went to a group."
  expect(prompt).toContain(`statement"):\n${ev}\n\nCONTEXT WINDOW`)
  expect(prompt).toContain("image descriptions):\n## Windows\nline\n")
  expect(prompt).toContain("ANSWER DIRECTIVE:\nAnswer carefully.")
  expect(prompt).not.toContain("hit text")
})

test("lossless: every non-blank context line except the Evidence header is rendered exactly once", () => {
  const items = mapShimReply(reply) as Array<Record<string, any>>
  const rendered = [
    ...items.filter((i) => i.kind === "kg_evidence_line").map((i) => i.text),
    ...items.find((i) => i.kind === "cxn_context")!.lines,
  ].filter((l: string) => l.trim() !== "")
  const source = reply.context.split("\n").filter((l) => l.trim() !== "" && l !== "## Evidence")
  expect(rendered.sort()).toEqual(source.sort())
})

test("no directive -> no directive block", () => {
  const prompt = buildKernelAnswerPrompt("When?", mapShimReply({ ...reply, directive: null }))
  expect(prompt).not.toContain("ANSWER DIRECTIVE")
})
