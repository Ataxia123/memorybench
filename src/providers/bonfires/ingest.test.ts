import { describe, it, expect } from "bun:test";
import { serializeSession } from "./ingest.js";

describe("ingest helpers", () => {
  it("serializeSession joins messages as [role]: content lines", () => {
    const out = serializeSession({
      sessionId: "s1",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
      ],
    });
    expect(out).toBe("[user]: hi\n[assistant]: hello");
  });

  it("serializeSession handles single-message sessions", () => {
    const out = serializeSession({
      sessionId: "s2",
      messages: [{ role: "user", content: "just this" }],
    });
    expect(out).toBe("[user]: just this");
  });
});
