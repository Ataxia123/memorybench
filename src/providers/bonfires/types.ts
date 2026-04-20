export interface DelveAgent {
  id: string
  name: string
  bonfireId: string
}

export interface StackMessage {
  id: string
  text: string
  userId: string
  chatId: string
  timestamp: string   // ISO 8601
  role: string        // "user" in whodunit; kept flexible
}

export interface JobStatus {
  state: "pending" | "running" | "completed" | "failed" | "cancelled"
  error?: string
  // Legacy top-level result — used by some callers. The actual workflow
  // output usually lives under metadata.result instead.
  result?: unknown
  metadata?: {
    job_uuid?: string
    result?: Record<string, unknown>
  }
}

export interface VectorSearchResult {
  text: string
  score: number | null
  id?: string
  doc_snippet?: string
}

export interface KgDelveEdge {
  fact: string
  score?: number
  uuid?: string
}

export interface KgDelveResult {
  edges?: KgDelveEdge[]
  nodes?: Array<{ name: string; summary?: string; uuid?: string }>
}

export type BonfiresArm = "vector" | "graph" | "smart"

export interface BonfiresConfig {
  apiUrl: string
  apiKey?: string
  arm: BonfiresArm
  bonfireId: string
}
