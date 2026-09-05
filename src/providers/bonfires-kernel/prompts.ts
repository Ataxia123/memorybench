import { buildAnswerPromptV3 } from "../bonfires-cxn/retrieval2"

/** Hide ingestion-time annotations without removing occurrence identity or stored provenance. */
export function buildKernelAnswerPrompt(
  question: string, context: unknown[], questionDate?: string
): string {
  return buildAnswerPromptV3(question, context, questionDate, { omitStatementCreatedAtAnnotation: true })
}
