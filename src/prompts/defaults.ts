import { buildContextString } from "../types/prompts"

export function buildDefaultAnswerPrompt(
  question: string,
  context: unknown[],
  questionDate?: string
): string {
  const contextStr = buildContextString(context)

  // MEMBENCH_PROMPT=v2 switches to the abstention-first prompt (same
  // rules, stronger wording on "I don't know"). Kept env-gated so
  // in-flight runs and historic baselines use the original v1 prompt.
  if (process.env.MEMBENCH_PROMPT === "v2") {
    return `You are a question-answering system. Based ONLY on the retrieved context below, answer the question.

Question: ${question}
Question Date: ${questionDate || "Not specified"}

Retrieved Context:
${contextStr}

Rules:
1. If the context does not clearly support an answer, respond "I don't know". Being cautious is correct — do not guess or infer beyond what the context directly states.
2. Only use information from the retrieved context. Do not use outside knowledge.
3. If a date is relevant, use the Question Date as reference and any event_time fields in the context to reason about timing.
4. Answer concisely — a single fact when possible, a short phrase otherwise.

Answer:`
  }

  return `You are a question-answering system. Based on the retrieved context below, answer the question.

Question: ${question}
Question Date: ${questionDate || "Not specified"}

Retrieved Context (raw JSON from memory provider):
${contextStr}

Instructions:
- The context above is the raw JSON response from a memory search API
- Extract relevant information from the JSON to answer the question
- Consider any temporal/date information present in the data
- If the context contains enough information, provide a clear, concise answer
- If the context does not contain enough information, respond with "I don't know"
- Base your answer ONLY on the provided context

Answer:`
}

export const DEFAULT_JUDGE_PROMPT = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer`

export const ABSTENTION_JUDGE_PROMPT = `You are evaluating an abstention question. The correct answer is that the information was NOT in the conversation, so the system should abstain or say it doesn't know.

The hypothesis is CORRECT if the system correctly abstains, says "I don't know", indicates uncertainty, or explicitly states the information is not available. It is INCORRECT if the system makes up an answer or hallucinates.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the system properly abstained
{"score": 0, "label": "incorrect", "explanation": "..."} if the system hallucinated an answer`

export const TEMPORAL_JUDGE_PROMPT = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer`

export const KNOWLEDGE_UPDATE_JUDGE_PROMPT = `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response contains the correct answer
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not contain the correct answer`

export const PREFERENCE_JUDGE_PROMPT = `I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Respond with ONLY a JSON object:
{"score": 1, "label": "correct", "explanation": "..."} if the response satisfies the rubric
{"score": 0, "label": "incorrect", "explanation": "..."} if the response does not satisfy the rubric`

export function getJudgePromptForType(questionType: string, groundTruth?: string): string {
  const type = questionType.toLowerCase()

  // Real-GT guard: LoCoMo mints groundTruth via `String(qa.answer)` (locomo/index.ts),
  // so an unanswerable adversarial question has groundTruth === "undefined" (no
  // `answer` field on the source item) while an adversarial question with a real
  // answer (e.g. "No") carries an actual string. ABSTENTION_JUDGE_PROMPT can only
  // score "did the system abstain" — routing a real-GT question through it makes a
  // correct concrete answer unscorable (q167/q178: answered "No" correctly, scored
  // incorrect because the judge only checks for abstention language). Only route to
  // the abstention prompt when there is no real ground truth to check the answer
  // against. `groundTruth` is optional here (callers that never had it to pass keep
  // their old type-only routing) — the guard only fires when a real value is present
  // and it is literally the "undefined" sentinel.
  const hasRealGroundTruth = groundTruth !== undefined && groundTruth !== "undefined"
  if ((type.includes("abstention") || type.includes("adversarial")) && !hasRealGroundTruth) {
    return ABSTENTION_JUDGE_PROMPT
  }

  if (type.includes("temporal")) {
    return TEMPORAL_JUDGE_PROMPT
  }

  if (type.includes("update") || type.includes("changing")) {
    return KNOWLEDGE_UPDATE_JUDGE_PROMPT
  }

  if (type.includes("preference")) {
    return PREFERENCE_JUDGE_PROMPT
  }

  return DEFAULT_JUDGE_PROMPT
}
