# rl_agent/evaluate.py
"""
Evaluator: TTL / LRU / LFU / DRL on a request trace.

Outputs per-policy metrics:
- total, hits, hit_ratio_pct
- avg_latency_ms, p95_latency_ms
- stale_pct (kept for symmetry; we don't serve stale here)
- bytes_from_cache, bytes_from_origin, bandwidth_savings_pct
- provenance: trace, model_path, max_bytes, ttl_s, hit_ms, seed

CLI:
  python -m rl_agent.evaluate --trace <file.csv> --model <path.pt> --out <report.csv> \
    --max-bytes 50000000 --ttl-s 300 --hit-ms 20 --seed 123 --plot --plot-save plot.png
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import pandas as pd

# -------------------------------
# Helpers: load trace and aliases
# -------------------------------
ALIASES = {
    "ts": ["ts", "timestamp", "time", "datetime", "request_ts"],
    "key": ["key", "cache_key", "object_id", "id", "item", "path", "url"],
    "size": ["size", "bytes", "content_length", "object_size", "object_size_bytes", "resp_bytes", "response_size"],
    "was_write": ["was_write", "write_flag", "is_write", "write", "op", "operation", "op_type", "method"],
}

def _coerce_ts(col: pd.Series) -> pd.Series:
    # numeric -> seconds; strings -> parse to datetime -> seconds
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").astype(float)
    dt = pd.to_datetime(col, errors="coerce", utc=True, infer_datetime_format=True)
    # convert ns->s; .view('int64') warnings on newer pandas -> use .astype('int64', errors='ignore') guard
    ns = dt.view("int64")
    return (ns / 1_000_000_000.0)

def _coerce_write(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        return (pd.to_numeric(col, errors="coerce") > 0).astype(int)
    low = col.astype(str).str.lower()
    return low.isin(["1", "true", "w", "write", "put", "post", "delete"]).astype(int)

def load_trace_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace not found: {path}")
    df = pd.read_csv(p)

    # alias resolution
    rename: Dict[str, str] = {}
    for canon, alist in ALIASES.items():
        if canon in df.columns:
            continue
        found = next((a for a in alist if a in df.columns), None)
        if not found:
            raise ValueError(f"Trace missing required column or alias for '{canon}' (aliases: {alist})")
        rename[found] = canon
    if rename:
        print(f"[evaluate] alias mapping: {rename}")
        df = df.rename(columns=rename)

    # type coercions
    df["ts"] = _coerce_ts(df["ts"])
    if df["ts"].isna().any():
        raise ValueError("Could not parse some timestamps to seconds.")
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["was_write"] = _coerce_write(df["was_write"]).astype(int)
    df["key"] = df["key"].astype(str)

    # sort by time just in case
    df = df.sort_values("ts", ascending=True).reset_index(drop=True)
    return df

# -------------------------------------
# Core simulator for TTL / LRU / LFU / DRL
# -------------------------------------
class CacheEntry:
    __slots__ = ("size", "expiry", "freq", "last_access")
    def __init__(self, size: int, expiry: float, ts: float):
        self.size = int(size)
        self.expiry = float(expiry)
        self.freq = 1
        self.last_access = float(ts)

def _evict_until_fit(cache: Dict[str, "CacheEntry"],
                     current_bytes: int,
                     need: int,
                     max_bytes: int,
                     policy: str) -> Tuple[int, Dict[str, "CacheEntry"]]:
    """Evict according to policy (lfu or lru) until there is room."""
    if max_bytes <= 0:
        return current_bytes, cache
    while current_bytes + need > max_bytes and cache:
        if policy == "lfu":
            # evict lowest freq, then oldest
            victim = min(cache.items(), key=lambda kv: (kv[1].freq, kv[1].last_access))[0]
        else:
            # lru: oldest last_access
            victim = min(cache.items(), key=lambda kv: kv[1].last_access)[0]
        current_bytes -= cache[victim].size
        cache.pop(victim, None)
    return current_bytes, cache

def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = int(math.ceil(0.95 * len(xs))) - 1
    k = max(0, min(k, len(xs) - 1))
    return float(xs[k])

def simulate_policy(df: pd.DataFrame,
                    policy: str,
                    max_bytes: int = 50_000_000,
                    ttl_s: int = 300,
                    hit_ms: int = 20,
                    agent: Any | None = None,
                    rng: random.Random | None = None) -> Dict[str, float]:
    """
    policy: "ttl" | "lru" | "lfu" | "drl"
    DRL uses agent.act([in_cache, current_bytes, size, remaining_ttl, was_write]).
    Cache-on-read; writes force refresh. Miss is served from origin, then cached.
    """
    cache: Dict[str, CacheEntry] = {}
    current_bytes = 0

    total = 0
    hits = 0
    stale_count = 0  # we don't serve stale; kept for symmetry
    sum_latency = 0.0
    per_latency: List[float] = []
    bytes_from_cache = 0
    bytes_from_origin = 0

    for row in df.itertuples(index=False):
        ts: float = float(row.ts)
        key: str = str(row.key)
        size: int = int(row.size)
        is_write: bool = bool(row.was_write)

        total += 1

        # check current cache state
        entry = cache.get(key)
        valid = False
        remaining = 0.0
        if entry is not None:
            valid = (ttl_s <= 0) or (entry.expiry >= ts)
            remaining = max(0.0, entry.expiry - ts)
            entry.last_access = ts
            entry.freq += 1

        # compute hit/miss BEFORE updates
        is_hit = bool(entry is not None and valid)
        if is_hit:
            hits += 1
            lat = float(hit_ms)
            bytes_from_cache += size
        else:
            # simple origin penalty (could be replaced with per-row latency if available)
            lat = float(hit_ms) * 5.0
            bytes_from_origin += size
        sum_latency += lat
        per_latency.append(lat)

        # decide whether to (re)cache/refresh
        do_refresh = False

        if policy in ("ttl", "lru", "lfu"):
            if is_write:
                do_refresh = True
            elif not is_hit:
                do_refresh = True  # cache-on-read

        elif policy == "drl":
            in_cache = 1.0 if entry is not None and entry.expiry >= ts else 0.0
            obs = [in_cache, float(current_bytes), float(size), float(remaining), 1.0 if is_write else 0.0]
            try:
                action = int(agent.act(obs, epsilon=0.0)) if agent is not None else 0
            except Exception:
                action = 0
            do_refresh = bool(action == 1 or is_write)

        # apply refresh if chosen
        if do_refresh:
            # ensure capacity
            current_bytes, cache = _evict_until_fit(
                cache, current_bytes, size, max_bytes,
                "lfu" if policy == "lfu" else "lru"
            )
            expiry = ts + float(ttl_s)
            if key not in cache:
                current_bytes += size
                cache[key] = CacheEntry(size=size, expiry=expiry, ts=ts)
            else:
                current_bytes += (size - cache[key].size)
                cache[key].size = size
                cache[key].expiry = expiry
                cache[key].last_access = ts

    avg_latency_ms = (sum_latency / total) if total > 0 else 0.0
    p95_latency_ms = _p95(per_latency)
    hit_ratio_pct = (100.0 * hits / total) if total > 0 else 0.0
    bandwidth_savings_pct = 0.0
    denom = bytes_from_cache + bytes_from_origin
    if denom > 0:
        bandwidth_savings_pct = 100.0 * (bytes_from_cache / float(denom))

    return {
        "policy": policy,
        "total": int(total),
        "hits": int(hits),
        "hit_ratio_pct": round(hit_ratio_pct, 4),
        "avg_latency_ms": round(avg_latency_ms, 4),
        "p95_latency_ms": round(p95_latency_ms, 4),
        "stale_pct": 0.0,
        "bytes_from_cache": int(bytes_from_cache),
        "bytes_from_origin": int(bytes_from_origin),
        "bandwidth_savings_pct": round(bandwidth_savings_pct, 4),
    }

# -------------------------
# DRL loader (optional)
# -------------------------
def load_agent(model_path: Optional[str]):
    if not model_path:
        return None
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        from rl_agent.dqn import DQNAgent
        return DQNAgent.load(str(mp), state_dim=5)
    except Exception as e:
        raise RuntimeError(f"Failed to load DRL agent: {e}")

# -------------------------
# Top-level API
# -------------------------
def evaluate_all(trace: str,
                 model_path: Optional[str],
                 max_bytes: int,
                 ttl_s: int,
                 hit_ms: int,
                 seed: Optional[int] = None) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    df = load_trace_csv(trace)
    rows: List[Dict[str, Any]] = []

    # TTL / LRU / LFU baselines
    rows.append(simulate_policy(df, "ttl", max_bytes=max_bytes, ttl_s=ttl_s, hit_ms=hit_ms))
    rows.append(simulate_policy(df, "lru", max_bytes=max_bytes, ttl_s=ttl_s, hit_ms=hit_ms))
    rows.append(simulate_policy(df, "lfu", max_bytes=max_bytes, ttl_s=ttl_s, hit_ms=hit_ms))

    # DRL (optional)
    if model_path:
        agent = load_agent(model_path)
        rows.append(simulate_policy(df, "drl", max_bytes=max_bytes, ttl_s=ttl_s, hit_ms=hit_ms, agent=agent))

    # Stamp provenance on each row
    for r in rows:
        r["trace"] = trace
        r["model_path"] = model_path or ""
        r["max_bytes"] = int(max_bytes)
        r["ttl_s"] = int(ttl_s)
        r["hit_ms"] = int(hit_ms)
        r["seed"] = seed if seed is not None else ""

    return rows

def save_report_csv(path: str, rows: List[Dict[str, Any]]):
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # fixed column order
    cols = [
        "policy",
        "total", "hits", "hit_ratio_pct",
        "avg_latency_ms", "p95_latency_ms",
        "stale_pct",
        "bytes_from_cache", "bytes_from_origin", "bandwidth_savings_pct",
        "trace", "model_path", "max_bytes", "ttl_s", "hit_ms", "seed",
    ]
    # include any extras without dropping data
    extra: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols and k not in extra:
                extra.append(k)
    pd.DataFrame(rows)[cols + extra].to_csv(out_p, index=False)

def plot_comparison(rows: List[Dict[str, Any]], save_path: Optional[str] = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    # order for consistency
    order = ["ttl", "lru", "lfu", "drl"]
    rows_sorted = [r for p in order for r in rows if r["policy"] == p]
    if not rows_sorted:
        return

    labels = [r["policy"].upper() for r in rows_sorted]
    hit = [r["hit_ratio_pct"] for r in rows_sorted]
    avg = [r["avg_latency_ms"] for r in rows_sorted]
    p95 = [r["p95_latency_ms"] for r in rows_sorted]
    bws = [r["bandwidth_savings_pct"] for r in rows_sorted]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.bar(labels, hit); ax1.set_title("Hit Ratio (%)")
    ax2.bar(labels, avg); ax2.set_title("Avg Latency (ms)")
    ax3.bar(labels, p95); ax3.set_title("p95 Latency (ms)")
    ax4.bar(labels, bws); ax4.set_title("Bandwidth Savings (%)")

    for ax, vals in [(ax1, hit), (ax2, avg), (ax3, p95), (ax4, bws)]:
        mx = max(vals) if any(vals) else 1.0
        for i, v in enumerate(vals):
            ax.text(i, v + (0.02 * mx), f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Caching Policy Comparison")
    fig.tight_layout()
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(base))
        print(f"Saved comparison plot to {base}")
    else:
        plt.show()

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="Path to simulator/external CSV trace")
    p.add_argument("--model", default=None, help="Path to DRL .pt (optional)")
    p.add_argument("--out", required=True, help="CSV output")
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    p.add_argument("--ttl-s", type=int, default=300)
    p.add_argument("--hit-ms", type=int, default=20)
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-save", default=None, help="If set, save comparison plot here")
    args = p.parse_args()

    rows = evaluate_all(args.trace, args.model, args.max_bytes, args.ttl_s, args.hit_ms, args.seed)
    save_report_csv(args.out, rows)
    if args.plot:
        plot_comparison(rows, args.plot_save)

if __name__ == "__main__":
    main()
