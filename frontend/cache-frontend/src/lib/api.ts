// frontend/src/lib/api.ts

/** ===== Shared rows for charts ===== */
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

/** ===== Top-level stats ===== */
export type Stats = {
  hit_ratio_pct: number;
  avg_latency_ms: number;
  staleness_pct: number;
};

/** ===== Cache items & stats ===== */
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

/** ===== Model registry types ===== */
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
  last_eval?: {
    ts: string;
    csv: string;
  };
};

/** ===== Evaluation CSV row ===== */
export type EvalRow = {
  policy: string;
  total: number;
  hits: number;
  hit_ratio_pct: number;
  avg_latency_ms: number;
  stale_pct: number;
  // Day 12+ extras (present if evaluator writes them):
  bytes_from_cache?: number;
  bytes_from_origin?: number;
  bandwidth_savings_pct?: number;
  p95_latency_ms?: number;
  // keep future-proof:
  [k: string]: number | string | undefined;
};

/** ===== Train / Evaluate response shapes ===== */
export type TrainResponse = {
  status: "started" | "error";
  trace: string;
  epochs: number;
  out_name?: string;
};

/**
 * Supports BOTH backend variants:
 * - legacy: { out: string, ttl: {...}, drl?: {...} }
 * - current: { status: "ok", eval_log: string }
 * Optional `rows` (if backend parses CSV inline).
 */
export type EvaluateResponse = {
  status?: string;
  out?: string;
  eval_log?: string;
  rows?: EvalRow[];
  ttl?: {
    total: number;
    hits: number;
    hit_ratio_pct: number;
    avg_latency_ms: number;
    stale_pct: number;
  };
  drl?:
    | {
        total: number;
        hits: number;
        hit_ratio_pct: number;
        avg_latency_ms: number;
        stale_pct: number;
      }
    | null;
};

/** ===== Experiments (start run) ===== */
export type StartRunPayload = {
  workload: string;
  minutes: number;
  rps: number;
  rate: number;
  objects?: number;
  clients?: number;
  seed?: number;
};

export type StartRunResponse = {
  run_id: number;
  status: string; // "started" | "error"
};

/** ===== Active / Promote model ===== */
export type ActiveModel = {
  path: string | null;
  loaded: boolean;
};

export type PromoteResponse = {
  status: "ok";
  active: string; // path to rl_agent/model.pt
  loaded: boolean;
};

/** ===== Small fetch helper ===== */
async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

/** ===== API surface ===== */
export const api = {
  // Dashboard
  stats: () => getJSON<Stats>("/api/stats"),
  cache: () => getJSON<CacheItem[]>("/api/cache"),
  cacheStats: () => getJSON<CacheStats>("/api/cache/stats"),
  history: (window = 120) => getJSON<HistoryRow[]>(`/api/history?window=${window}`),
  
  historyByRun: (runId: number, window = 120) =>
    getJSON<HistoryRow[]>(`/api/history?run_id=${runId}&window=${window}`),

  runs: () => getJSON<RunRow[]>("/api/runs"),

  // Models
  models: () => getJSON<ModelEntry[]>("/api/models"),

  trainModel: async (payload: { trace?: string; epochs?: number }): Promise<TrainResponse> => {
    const res = await fetch(`${BASE}/api/models/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<TrainResponse>;
  },

  evaluateModel: async (payload: {
    model_id?: string;
    model_path?: string;
    trace?: string;
    out?: string;
    plot?: boolean;
    max_bytes?: number;   
    ttl_s?: number;       
    hit_ms?: number; 
  }): Promise<EvaluateResponse> => {
    const res = await fetch(`${BASE}/api/models/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<EvaluateResponse>;
  },

  getModelReport: async (model_id: string) => {
    const res = await fetch(`${BASE}/api/models/report?model_id=${encodeURIComponent(model_id)}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<{ model: ModelEntry; csv: string; ts: string; rows: EvalRow[] }>;
  },

  resetModelRegistry: async () => {
    const res = await fetch(`${BASE}/api/models/registry/reset`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  activeModel: async (): Promise<ActiveModel> => {
    const res = await fetch(`${BASE}/api/models/active`);
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<ActiveModel>;
  },

  promoteModel: async (payload: { model_id?: string; model_path?: string }): Promise<PromoteResponse> => {
    const res = await fetch(`${BASE}/api/models/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<PromoteResponse>;
  },

  // Experiments
  startRun: async (payload: StartRunPayload): Promise<StartRunResponse> => {
    const res = await fetch(`${BASE}/api/experiments/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<StartRunResponse>;
  },

  // Simple simulator trigger
  simulate: (mode: string) =>
    fetch(`${BASE}/api/simulate?mode=${encodeURIComponent(mode)}`, {
      method: "POST",
    }).then(async (r) => {
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    }),
};

/** Named exports (so components can cherry-pick) */
export const {
  stats,
  cache,
  cacheStats,
  history,
  historyByRun,
  runs,
  models,
  simulate,
  trainModel,
  evaluateModel,
  getModelReport,
  resetModelRegistry,
  activeModel,
  promoteModel,
  startRun,
} = api;
