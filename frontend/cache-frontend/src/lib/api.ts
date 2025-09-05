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

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  stats: () => getJSON<Stats>("/api/stats"),
  cache: () => getJSON<CacheItem[]>("/api/cache"),
  cacheStats: () => getJSON<CacheStats>("/api/cache/stats"),
};
