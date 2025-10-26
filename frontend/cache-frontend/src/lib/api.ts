// frontend/src/lib/api.ts

/** ---------- Types for shared UI ---------- */
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

/** ---------- Model registry / evaluation types ---------- */
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
  // Backend may add last_eval later; keep it optional/flexible
  last_eval?: {
    ts: string;
    csv: string;
  };
};

export type TrainResponse = {
  status: "started" | "error";
  trace: string;
  epochs: number;
  out_name?: string;
};

/** NOTE: This matches your backend main.py:
 *   return {"status": "ok", "eval_log": out_log}
 */
export type EvaluateResponse = {
  status: "ok";
  eval_log: string;
};

/** When reading parsed report rows via /api/models/report */
export type EvalRow = {
  policy: string;
  total: number;
  hits: number;
  hit_ratio_pct: number;
  avg_latency_ms: number;
  stale_pct: number;
};

export type ReportResponse = {
  csv: string;
  ts: string;
  rows: EvalRow[];
  model?: ModelEntry;
};

/** ---------- HTTP helpers ---------- */
const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

/** ---------- API surface ---------- */
export const api = {
  stats: () => getJSON<Stats>("/api/stats"),
  cache: () => getJSON<CacheItem[]>("/api/cache"),
  cacheStats: () => getJSON<CacheStats>("/api/cache/stats"),
  history: (window = 120) => getJSON<HistoryRow[]>(`/api/history?window=${window}`),
  runs: () => getJSON<RunRow[]>("/api/runs"),
  models: () => getJSON<ModelEntry[]>("/api/models"),

  simulate: (mode: "ttl" | "drl" | "hybrid") =>
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

  // Evaluate a model (synchronous) — returns only where the CSV was written
  evaluateModel: async (payload: {
    path?: string;
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

  // Read parsed CSV rows for display
  getModelReport: async (model_id: string) =>
    getJSON<ReportResponse>(`/api/models/report?model_id=${encodeURIComponent(model_id)}`),

  // Clear registry.json (frontend "Available Models" list)
  resetModelRegistry: async () => {
    const res = await fetch(`${BASE}/api/models/registry/reset`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<{ status: "ok"; message: string }>;
  },
};

/** Convenience named exports */
export const {
  stats,
  cache,
  cacheStats,
  history,
  runs,
  models,
  simulate,
  trainModel,
  evaluateModel,
  getModelReport,
  resetModelRegistry,
} = api;

/** Experiments runner */
export const startRun = async (payload: { workload: string; minutes: number; rps: number; rate: number }) => {
  const res = await fetch(`${BASE}/api/experiments/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

export const getActiveModel = async (): Promise<{ path: string | null; loaded: boolean }> => {
  const res = await fetch(`${BASE}/api/models/active`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

export const promoteModel = async (payload: { model_id?: string; model_path?: string }) => {
  const res = await fetch(`${BASE}/api/models/promote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};
