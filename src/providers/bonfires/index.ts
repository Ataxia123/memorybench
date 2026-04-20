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
import { ingestSessions } from "./ingest.js"
import { runIndexingPipeline } from "./indexing.js"
import { armSearch } from "./search.js"

export class BonfiresProvider implements Provider {
  name = "bonfires"
  private client!: BonfiresClient
  private config!: BonfiresConfig
  private agentId!: string

  async initialize(config: ProviderConfig): Promise<void> {
    // getProviderConfig("bonfires") returns { apiKey, apiUrl, arm, bonfireId }
    // via ProviderConfig's [key: string]: unknown index signature.
    const apiUrl = config.apiUrl as string | undefined
    const arm = config.arm as BonfiresConfig["arm"] | undefined
    const bonfireId = config.bonfireId as string | undefined

    if (!apiUrl || !arm || !bonfireId) {
      throw new Error("bonfires provider requires { apiUrl, arm, bonfireId } in config")
    }

    this.config = {
      apiUrl,
      apiKey: config.apiKey || undefined,
      arm,
      bonfireId,
    }

    this.client = new BonfiresClient({ apiUrl: this.config.apiUrl, apiKey: this.config.apiKey })
    await this.client.healthz()

    // Ensure Weaviate has the Bonfire_labels / Owl_classes collections.
    // Idempotent — safe to call on every run. Required before update_labels.
    await this.client.setupVectorStore()

    // Bootstrap the bonfire document in MongoDB before creating the agent,
    // so the agent's bonfireId reference is valid.
    // ensureBonfire returns the resolved 24-char hex ObjectId (slug → SHA-1 if needed).
    const originalSlug = this.config.bonfireId
    const resolvedBonfireId = await this.client.ensureBonfire({
      bonfireId: originalSlug,
      name: `memorybench-${originalSlug}`,
      primaryGrammar: "locomo",
    })
    // Always use the hex form for all subsequent API calls (Delve requires a valid ObjectId).
    this.config = { ...this.config, bonfireId: resolvedBonfireId }

    const agent = await this.client.findOrCreateAgent({
      bonfireId: resolvedBonfireId,
      name: `memorybench-${originalSlug}`,
    })
    this.agentId = agent.id
  }

  async ingest(sessions: UnifiedSession[], _options: IngestOptions): Promise<IngestResult> {
    return ingestSessions({ client: this.client, agentId: this.agentId, bonfireId: this.config.bonfireId, sessions })
  }

  async awaitIndexing(
    _result: IngestResult,
    _containerTag: string,
    _onProgress?: IndexingProgressCallback
  ): Promise<void> {
    await runIndexingPipeline({
      client: this.client,
      agentId: this.agentId,
      bonfireId: this.config.bonfireId,
    })
  }

  async search(query: string, _options: SearchOptions): Promise<unknown[]> {
    return armSearch({ client: this.client, query, config: this.config })
  }

  async clear(_containerTag: string): Promise<void> {
    // local-only setup: no-op. Users drop state manually between runs.
  }
}
