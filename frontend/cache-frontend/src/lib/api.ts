// frontend/src/lib/api.ts
export type HistoryRow = {
  ts: string;
  hit_ratio_pct: number;
  avg_latency_ms: number;
  staleness_pct: number;
};

export type RunRow = {
  id: number;
  started_ts: string;
  workload: string;
  minutes: number;
  rps: number;
  rate: number;
  status: string;
};

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type Stats = {
  hit_ratio_pct: number;
  avg_latency_ms: number;
  staleness_pct: number;
};

export type CacheItem = {
  object_id: string;
  size_bytes: number;
  last_updated_ts: string;
  ttl_s: number;
};

export type CacheStats = {
  items: number;
  bytes_used: number;
  max_bytes: number;
  pct_full: number;
};

/** Model registry types */
export type ModelMeta = {
  trace?: string;
  epochs?: number;
  [key: string]: unknown;
};

export type ModelEntry = {
  id: string;
  path: string;
  created_ts: string;
  meta?: ModelMeta;
};

/** Responses for training/evaluation endpoints */
export type TrainResponse = {
  status: "started" | "error";
  trace: string;
  epochs: number;
  out_name?: string;
};

export type EvaluateResponse = {
  out: string;
  ttl: {
    total: number;
    hits: number;
    hit_ratio_pct: number;
    avg_latency_ms: number;
    stale_pct: number;
  };
  drl?: {
    total: number;
    hits: number;
    hit_ratio_pct: number;
    avg_latency_ms: number;
    stale_pct: number;
  } | null;
};

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  stats: () => getJSON<Stats>("/api/stats"),
  cache: () => getJSON<CacheItem[]>("/api/cache"),
  cacheStats: () => getJSON<CacheStats>("/api/cache/stats"),
  history: (window = 120) => getJSON<HistoryRow[]>(`/api/history?window=${window}`),
  runs: () => getJSON<RunRow[]>("/api/runs"),
  models: () => getJSON<ModelEntry[]>("/api/models"),
  simulate: (mode: string) =>
    fetch(`${BASE}/api/simulate?mode=${encodeURIComponent(mode)}`, {
      method: "POST",
    }).then(async (r) => {
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }),

  // Start a background training job
  trainModel: async (payload: { trace?: string; epochs?: number }): Promise<TrainResponse> => {
    const res = await fetch(`${BASE}/api/models/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<TrainResponse>;
  },

  // Evaluate a model (synchronous)
  evaluateModel: async (payload: {
    model_id?: string;
    model_path?: string;
    trace?: string;
    out?: string;
    plot?: boolean;
  }): Promise<EvaluateResponse> => {
    const res = await fetch(`${BASE}/api/models/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<EvaluateResponse>;
  },
};

// Convenience named exports for direct imports
export const { stats, cache, cacheStats, history, runs, models, simulate, trainModel, evaluateModel } = api;
