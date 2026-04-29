import type { UnifiedMessage, UnifiedSession } from "../../types/unified.js"
import type { BonfiresClient } from "./client.js"

/** Dialogue transcript for the whole session — kept for reference but not
 * used now that ingest is per-message. */
export function serializeSession(session: {
  sessionId: string
  messages: Array<{ role: string; content: string; speaker?: string }>
}): string {
  return session.messages.map((m) => `[${m.speaker ?? m.role}]: ${m.content}`).join("\n")
}

function serializeMessage(m: UnifiedMessage): string {
  return `${m.speaker ?? m.role}: ${m.content}`
}

/**
 * Session-aware ingest: one `ingestContent` call per session, sending
 * the full structured `messages` payload. Delve's session-aware branch
 * then emits engram's 4-doc pattern against a single Document:
 *
 *   - session × N     (per-message, speaker-prefixed, GLiNER entities inline)
 *   - speaker × K     (per-speaker concatenated turns)
 *   - preference × K  (per-speaker GLiNER Preference/Emotion/Activity/Object aggregate)
 *   - topic × ≤1      (session-level Topic/Organization/Event/Location aggregate)
 *
 * All chunks share `doc_ref`. GLiNER2 runs inline during ingest (not
 * deferred to backfill), so entity metadata is available before any
 * downstream indexing step.
 *
 * Graph ingestion (stackAdd + stackProcess / createEpisodeDirect) stays
 * in `runIndexingPipeline` — it runs after the Ontology doc exists so
 * graphiti's entity extraction gets ontology_entity_types guidance.
 */
export async function ingestSessions(args: {
  client: BonfiresClient
  agentId: string
  bonfireId: string
  sessions: UnifiedSession[]
}): Promise<{ documentIds: string[] }> {
  const { client, bonfireId, sessions } = args
  const documentIds: string[] = []

  for (const session of sessions) {
    const sessionDate = session.metadata?.date as string | undefined
    const documentContent = session.messages.map(serializeMessage).join("\n")
    const messagesPayload = session.messages.map((m, i) => ({
      speaker: m.speaker ?? m.role,
      content: m.content,
      timestamp: m.timestamp ?? sessionDate ?? null,
      role: m.role,
      msg_id: `m${i}`,
    }))

    await client.ingestContent({
      bonfireId,
      content: documentContent,
      title: `locomo-${session.sessionId}`,
      metadata: {
        session_id: session.sessionId,
        timestamp: sessionDate ?? null,
      },
      messages: messagesPayload,
    })
    documentIds.push(session.sessionId)
  }

  return { documentIds }
}
