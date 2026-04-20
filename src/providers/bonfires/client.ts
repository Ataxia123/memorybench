import type { StackMessage, JobStatus, VectorSearchResult, KgDelveResult } from "./types.js";

type FetchLike = typeof fetch;

export interface BonfiresClientOptions {
  apiUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  fetchImpl?: FetchLike;
}

export class BonfiresClient {
  private apiUrl: string;
  private apiKey?: string;
  private timeoutMs: number;
  private fetchImpl: FetchLike;

  constructor(opts: BonfiresClientOptions) {
    this.apiUrl = opts.apiUrl.replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 600_000;
    this.fetchImpl = opts.fetchImpl ?? fetch;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) {
      h.Authorization = `Bearer ${this.apiKey}`;
      h["X-API-Key"] = this.apiKey;
    }
    return h;
  }

  private async req<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
    const r = await this.fetchImpl(`${this.apiUrl}${path}`, {
      method,
      headers: this.headers(),
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`${method} ${path} failed ${r.status}: ${text}`);
    }
    return (await r.json()) as T;
  }

  healthz(): Promise<{ status: string }> {
    return this.req("GET", "/healthz");
  }

  createAgent(args: { bonfireId: string; name: string }): Promise<{ id: string; name: string }> {
    return this.req("POST", "/agents", {
      name: args.name,
      username: args.name,
      bonfireId: args.bonfireId,
      isActive: true,
    });
  }

  async findOrCreateAgent(args: { bonfireId: string; name: string }): Promise<{ id: string; name: string }> {
    try {
      return await this.createAgent(args);
    } catch (err) {
      const msg = String(err);
      if (!msg.includes("409")) throw err;
      const list = await this.req<{ agents: Array<{ id: string; name: string }> }>("GET", "/agents");
      const match = list.agents.find((a) => a.name === args.name);
      if (!match) throw new Error(`409 on createAgent but no existing agent '${args.name}' found`);
      return match;
    }
  }

  stackAdd(agentId: string, messages: StackMessage[]): Promise<unknown> {
    return this.req("POST", `/agents/${agentId}/stack/add`, { messages });
  }

  stackProcess(agentId: string): Promise<{ task_id: string }> {
    return this.req("POST", `/agents/${agentId}/stack/process`);
  }

  jobStatus(jobId: string): Promise<JobStatus> {
    return this.req("GET", `/jobs/${jobId}/status`);
  }

  async waitForJob(
    jobId: string,
    opts: { kind: string; timeoutSec?: number; pollIntervalSec?: number } = { kind: "job" },
  ): Promise<JobStatus> {
    const timeout = opts.timeoutSec ?? 3000;
    const poll = opts.pollIntervalSec ?? 5;
    const start = Date.now();
    await new Promise((r) => setTimeout(r, 2000));
    while (true) {
      const status = await this.jobStatus(jobId);
      if (status.state === "completed") return status;
      if (status.state === "failed" || status.state === "cancelled") {
        throw new Error(`${opts.kind} job ${jobId} ended in ${status.state}: ${status.error ?? ""}`);
      }
      if ((Date.now() - start) / 1000 > timeout) {
        throw new Error(`${opts.kind} job ${jobId} exceeded ${timeout}s`);
      }
      await new Promise((r) => setTimeout(r, poll * 1000));
    }
  }

  startTaxonomy(bonfireId: string): Promise<{ job_id: string }> {
    return this.req("POST", "/trigger_taxonomy", { bonfire_id: bonfireId });
  }

  startLabeling(bonfireId: string, vectorThreshold = 0.7): Promise<{ job_id: string }> {
    return this.req("POST", "/labeling/hybrid", {
      bonfire_id: bonfireId,
      is_multi_label: false,
      taxonomy_run_id: null,
      vector_threshold: vectorThreshold,
    });
  }

  buildGrammar(bonfireId: string, dryRun = false): Promise<unknown> {
    return this.req("POST", `/trimtabs/grammars/${bonfireId}/build`, { dry_run: dryRun });
  }

  buildOntology(
    bonfireId: string,
    opts: { linkToGraph?: boolean } = {},
  ): Promise<unknown> {
    return this.req("POST", `/ontology/${bonfireId}/build`, {
      entity_labels: null,
      link_to_graph: opts.linkToGraph ?? true,
      threshold: null,
      top_n_cap: null,
    });
  }

  async buildCommunities(bonfireId: string, sampleSize = 10): Promise<unknown> {
    const url = `${this.apiUrl}/knowledge_graph/communities/build?bonfire_id=${encodeURIComponent(
      bonfireId,
    )}&sample_size=${sampleSize}`;
    const r = await this.fetchImpl(url, { method: "POST", headers: this.headers() });
    if (!r.ok) throw new Error(`buildCommunities failed ${r.status}`);
    return r.json();
  }

  async vectorSearch(args: { bonfireId: string; query: string; limit: number }): Promise<VectorSearchResult[]> {
    const raw = await this.req<{
      results: Array<{ id: string; properties: Record<string, unknown>; score: number | null }>;
    }>("POST", "/vector_store/search", {
      bonfire_id: args.bonfireId,
      search_string: args.query,
      limit: args.limit,
    });
    return raw.results.map((r) => {
      const props = (r.properties ?? {}) as Record<string, unknown>;
      return {
        id: r.id,
        text: (props.content as string | undefined) ?? "",
        doc_snippet: props.doc_snippet as string | undefined,
        score: r.score,
      };
    });
  }

  kgDelve(args: {
    bonfireId: string;
    query: string;
    numResults: number;
    centerNodeUuid?: string;
    smart?: boolean;
    autoResolveCenter?: boolean;
    searchRecipe?: string;
  }): Promise<KgDelveResult> {
    const body: Record<string, unknown> = {
      bonfire_id: args.bonfireId,
      query: args.query,
      num_results: args.numResults,
      auto_resolve_center: args.autoResolveCenter ?? true,
    };
    if (args.centerNodeUuid) body.center_node_uuid = args.centerNodeUuid;
    if (args.smart) body.smart = true;
    if (args.searchRecipe) body.search_recipe = args.searchRecipe;
    return this.req("POST", "/delve", body);
  }
}
