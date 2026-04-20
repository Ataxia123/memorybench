import type { UnifiedSession } from "../../types/unified.js";
import type { BonfiresClient } from "./client.js";

/** Serialize a session as a dialogue transcript for ingest_content + stack_add. */
export function serializeSession(session: {
  sessionId: string;
  messages: Array<{ role: string; content: string }>;
}): string {
  return session.messages.map((m) => `[${m.role}]: ${m.content}`).join("\n");
}

/**
 * Ingest sessions into the Bonfires stack using whodunit's episodic semantics.
 *
 * For each session:
 *   1. ingest_content  — builds the vector store
 *   2. stack_add (one message = whole transcript) + stack_process + waitForJob
 *      → each session becomes one KG episode
 */
export async function ingestSessions(args: {
  client: BonfiresClient;
  agentId: string;
  bonfireId: string;
  sessions: UnifiedSession[];
}): Promise<{ documentIds: string[] }> {
  const { client, agentId, bonfireId, sessions } = args;
  const documentIds: string[] = [];

  for (const session of sessions as unknown as Array<{
    sessionId: string;
    messages: Array<{ role: string; content: string }>;
  }>) {
    const transcript = serializeSession(session);

    // Vector store path
    await client.ingestContent({
      bonfireId,
      content: transcript,
      title: `locomo-${session.sessionId}`,
    });

    // KG path: one whodunit-style stack message → stack_process → wait (one episode per session)
    const msg = {
      id: `${session.sessionId}-0`,
      text: transcript,
      userId: agentId,
      chatId: bonfireId,
      timestamp: new Date().toISOString(),
      role: "user",
    };
    await client.stackAdd(agentId, [msg]);
    const { task_id } = await client.stackProcess(agentId);
    await client.waitForJob(task_id, { kind: "stack_processing", timeoutSec: 1800 });

    documentIds.push(session.sessionId);
  }

  return { documentIds };
}
