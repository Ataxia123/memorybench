import { describe, it, expect } from "bun:test";
import { chunkMessages, formatSessionMessages } from "./ingest.js";

describe("ingest helpers", () => {
  it("formatSessionMessages converts UnifiedMessage to StackMessage", () => {
    const out = formatSessionMessages([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
    expect(out).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
  });

  it("chunkMessages splits into size-2 batches", () => {
    const msgs = [
      { role: "user" as const, content: "1" },
      { role: "assistant" as const, content: "2" },
      { role: "user" as const, content: "3" },
    ];
    const batches = chunkMessages(msgs, 2);
    expect(batches).toHaveLength(2);
    expect(batches[0]).toHaveLength(2);
    expect(batches[1]).toHaveLength(1);
  });
});
