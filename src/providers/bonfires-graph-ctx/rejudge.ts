// Re-judge stored answers with a chosen judge prompt. usage: bun rejudge.ts <report.json> <strict|lenient> <out.json>
import { readFileSync, writeFileSync } from "node:fs"
import { createJudge } from "../../judges"
import { getJudgeConfig } from "../../utils/config"
import { buildZepJudgePrompt } from "../zep/prompts"

const [reportPath, mode, outPath] = process.argv.slice(2)
const ev = JSON.parse(readFileSync(reportPath, "utf8")).evaluations as any[]
const judge = createJudge("openai")
const cfg = getJudgeConfig("openai")
;(cfg as any).model = "gpt-4.1-mini"
await judge.initialize(cfg)
const providerPrompts = mode === "lenient" ? ({ judgePrompt: buildZepJudgePrompt } as any) : undefined
const out: any[] = new Array(ev.length)
let next = 0
async function worker() {
  while (next < ev.length) {
    const i = next++
    const e = ev[i]
    const r = await judge.evaluate({
      runId: "rejudge", question: e.question, questionType: e.questionType,
      groundTruth: e.groundTruth, hypothesis: e.hypothesis, providerPrompts,
    })
    out[i] = { questionId: e.questionId, questionType: e.questionType, orig_score: e.score, score: r.score, label: r.label, explanation: r.explanation }
  }
}
await Promise.all([1, 2, 3, 4, 5].map(worker))
writeFileSync(outPath, JSON.stringify(out, null, 1), { flag: "wx" })
console.log("done", out.length)
