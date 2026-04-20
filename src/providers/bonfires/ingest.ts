import type { UnifiedMessage, UnifiedSession } from "../../types/unified.js";
import type { BonfiresClient } from "./client.js";
import type { StackMessage } from "./types.js";

export function formatSessionMessages(
  messages: UnifiedMessage[],
): StackMessage[] {
  return messages.map((m) => ({ role: m.role, content: m.content }));
}

export function chunkMessages(msgs: StackMessage[], size: number): StackMessage[][] {
  const out: StackMessage[][] = [];
  for (let i = 0; i < msgs.length; i += size) {
    out.push(msgs.slice(i, i + size));
  }
  return out;
}

/**
 * Ingest every session's messages onto the shared agent stack in 2-message batches.
 * No per-session stack_process — Task 5's indexing phase runs it once at the end.
 */
export async function ingestSessions(args: {
  client: BonfiresClient;
  agentId: string;
  sessions: UnifiedSession[];
}): Promise<{ documentIds: string[] }> {
  const documentIds: string[] = [];
  for (const session of args.sessions) {
    const messages = formatSessionMessages(session.messages);
    for (const batch of chunkMessages(messages, 2)) {
      await args.client.stackAdd(args.agentId, batch);
    }
    documentIds.push(session.sessionId);
  }
  return { documentIds };
}
