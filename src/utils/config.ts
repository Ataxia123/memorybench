export interface Config {
  supermemoryApiKey: string
  supermemoryBaseUrl: string
  mem0ApiKey: string
  zepApiKey: string
  openaiApiKey: string
  anthropicApiKey: string
  googleApiKey: string
  bonfiresApiUrl: string
  bonfiresApiKey: string
  bonfiresArm:
    | "vector"
    | "graph"
    | "smart_graph"
    | "smart"
    | "smart_full"
    | "smart_naked"
    | "smart_chunks_only"
    | "zep"
  bonfiresBonfireId: string
}

export const config: Config = {
  supermemoryApiKey: process.env.SUPERMEMORY_API_KEY || "",
  supermemoryBaseUrl: process.env.SUPERMEMORY_BASE_URL || "https://api.supermemory.ai",
  mem0ApiKey: process.env.MEM0_API_KEY || "",
  zepApiKey: process.env.ZEP_API_KEY || "",
  openaiApiKey: process.env.OPENAI_API_KEY || "",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY || "",
  googleApiKey: process.env.GOOGLE_API_KEY || "",
  bonfiresApiUrl: process.env.BONFIRES_API_URL || "http://localhost:8000",
  bonfiresApiKey: process.env.BONFIRES_API_KEY || "",
  bonfiresArm: (process.env.BONFIRES_ARM ?? "smart") as
    | "vector"
    | "graph"
    | "smart_graph"
    | "smart"
    | "smart_full"
    | "smart_naked"
    | "smart_chunks_only"
    | "zep",
  bonfiresBonfireId: process.env.BONFIRES_BONFIRE_ID || "locomo-eval",
}

export function getProviderConfig(provider: string): { apiKey: string; baseUrl?: string; [key: string]: unknown } {
  switch (provider) {
    case "supermemory":
      return { apiKey: config.supermemoryApiKey, baseUrl: config.supermemoryBaseUrl }
    case "mem0":
      return { apiKey: config.mem0ApiKey }
    case "zep":
      return { apiKey: config.zepApiKey }
    case "filesystem":
      return { apiKey: config.openaiApiKey } // Filesystem uses OpenAI for memory extraction
    case "rag":
      return { apiKey: config.openaiApiKey } // RAG provider uses OpenAI for embeddings
    case "bonfires":
      return {
        apiKey: config.bonfiresApiKey,
        apiUrl: config.bonfiresApiUrl,
        arm: config.bonfiresArm,
        bonfireId: config.bonfiresBonfireId,
      }
    default:
      throw new Error(`Unknown provider: ${provider}`)
  }
}

export function getJudgeConfig(judge: string): { apiKey: string; model?: string } {
  switch (judge) {
    case "openai":
      return { apiKey: config.openaiApiKey }
    case "anthropic":
      return { apiKey: config.anthropicApiKey }
    case "google":
      return { apiKey: config.googleApiKey }
    default:
      throw new Error(`Unknown judge: ${judge}`)
  }
}
