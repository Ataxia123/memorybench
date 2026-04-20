import type {
  Provider,
  ProviderConfig,
  IngestOptions,
  IngestResult,
  SearchOptions,
  IndexingProgressCallback,
} from "../../types/provider.js"
import type { UnifiedSession } from "../../types/unified.js"
import { BonfiresClient } from "./client.js"
import type { BonfiresConfig } from "./types.js"

export class BonfiresProvider implements Provider {
  name = "bonfires"
  private client!: BonfiresClient
  private config!: BonfiresConfig
  private agentId!: string

  async initialize(config: ProviderConfig): Promise<void> {
    throw new Error("not implemented")
  }

  async ingest(sessions: UnifiedSession[], options: IngestOptions): Promise<IngestResult> {
    throw new Error("not implemented")
  }

  async awaitIndexing(
    result: IngestResult,
    containerTag: string,
    onProgress?: IndexingProgressCallback
  ): Promise<void> {
    throw new Error("not implemented")
  }

  async search(query: string, options: SearchOptions): Promise<unknown[]> {
    throw new Error("not implemented")
  }

  async clear(containerTag: string): Promise<void> {
    // local-only setup: no-op. Users drop state manually between runs.
  }
}
