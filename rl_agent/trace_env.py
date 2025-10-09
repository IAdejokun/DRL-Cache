# rl_agent/trace_env.py
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import datetime, timezone
import random
from typing import List, Dict, Any

class TraceEnv:
    """
    Replay environment driven by a CSV trace (simulator output).
    Observations: [key_norm, in_cache(0/1), age_norm, cache_fill_frac, size_norm]
    Actions: 0 -> no cache/refresh, 1 -> cache/refresh
    Handles writes: increments origin_version for that object.
    """

    def __init__(
        self,
        csv_path: str,
        max_bytes: int = 50_000_000,
        ttl_s: int = 300,
        hit_latency_ms: int = 20,
        seed: int | None = None,
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # normalize column names to lower
        self.df.columns = [c.lower() for c in self.df.columns]
        # required columns mapping
        req_cols = ["ts", "client_id", "object_id", "object_size_bytes", "origin_latency_ms", "was_write"]
        for c in req_cols:
            if c not in self.df.columns:
                raise ValueError(f"Trace missing required column: {c}")
        # Convert was_write to bool
        self.df["was_write"] = self.df["was_write"].astype(str).str.lower().map(lambda v: v in ("1", "true", "t", "yes"))
        self.df = self.df.reset_index(drop=True)
        self.index = 0
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self.hit_latency_ms = hit_latency_ms
        self.seed = seed
        self.rng = random.Random(seed)

        # build object id -> numeric index map for normalization
        unique_ids = list(self.df["object_id"].unique())
        self.key_to_idx = {k: i+1 for i, k in enumerate(unique_ids)}  # 1..N
        self.num_keys = max(1, len(unique_ids))

        # origin versions: object_id -> int (increment on writes)
        self.origin_version = {k: 0 for k in unique_ids}

        # in-memory cache: object_id -> dict with inserted_ts, size_bytes, origin_version_at_store, last_access_ts
        self.cache: "OrderedDict[str, Dict[str,Any]]" = OrderedDict()
        self.bytes_used = 0

    def reset(self):
        self.index = 0
        self.cache.clear()
        self.bytes_used = 0
        # return first obs
        return self._make_obs_for_row(self.df.iloc[self.index])

    def _make_obs_for_row(self, row):
        object_id = row["object_id"]
        idx = self.key_to_idx.get(object_id, 1)
        key_norm = idx / float(self.num_keys)
        entry = self.cache.get(object_id)
        in_cache = 0
        age_norm = 1.0
        size_bytes = int(row["object_size_bytes"]) if not np.isnan(row["object_size_bytes"]) else 1024
        if entry:
            in_cache = 1
            age_s = (datetime.now(timezone.utc) - entry["inserted_ts"]).total_seconds()
            age_norm = min(age_s / float(self.ttl_s), 1.0)
            size_bytes = int(entry["size_bytes"])
        fill_frac = min(self.bytes_used / float(self.max_bytes), 1.0) if self.max_bytes else 0.0
        obs = [key_norm, float(in_cache), float(age_norm), float(fill_frac), float(size_bytes) / float(self.max_bytes or 1)]
        meta = {"object_id": object_id, "size_bytes": int(size_bytes), "row": row}
        return np.array(obs, dtype=np.float32), meta

    def _evict_until_fit(self, size_needed: int):
        while self.bytes_used + size_needed > self.max_bytes and len(self.cache) > 0:
            k, v = self.cache.popitem(last=False)
            self.bytes_used -= int(v["size_bytes"])

    def step(self, action: int):
        """
        Process current trace row using action.
        Returns: next_obs, reward, done, info
        """
        if self.index >= len(self.df):
            return None, 0.0, True, {}

        row = self.df.iloc[self.index]
        object_id = row["object_id"]
        size_bytes = int(row["object_size_bytes"])
        was_write = bool(row["was_write"])
        origin_latency = float(row["origin_latency_ms"]) if not np.isnan(row["origin_latency_ms"]) else float(self.hit_latency_ms)

        # update origin on write BEFORE serving (origin changes)
        if was_write:
            self.origin_version[object_id] = self.origin_version.get(object_id, 0) + 1

        entry = self.cache.get(object_id)
        in_cache = False
        cached_version = None
        age_s = None
        if entry:
            # check TTL
            age_s = (datetime.now(timezone.utc) - entry["inserted_ts"]).total_seconds()
            in_cache = age_s <= self.ttl_s
            cached_version = entry["origin_version"]

        # determine served_latency and stale
        served_latency = None
        stale = False
        served_from_cache = False

        # If action == 1 -> force cache/refresh (agent decided to cache)
        if action == 1:
            # evict if needed
            self._evict_until_fit(size_bytes)
            # insert or refresh with current origin version
            self.cache.pop(object_id, None)
            self.cache[object_id] = {"inserted_ts": datetime.now(timezone.utc), "size_bytes": size_bytes, "origin_version": self.origin_version.get(object_id, 0), "last_access_ts": datetime.now(timezone.utc)}
            self.bytes_used += size_bytes
            served_latency = float(self.hit_latency_ms)
            served_from_cache = True
            # fresh if inserted now, stale False
            stale = False
        else:
            # action == 0, do not actively cache; serve from cache if valid, else origin
            if in_cache and cached_version == self.origin_version.get(object_id, 0):
                # fresh cache hit
                served_latency = float(self.hit_latency_ms)
                served_from_cache = True
                stale = False
                # touch for LRU
                e = self.cache.pop(object_id)
                e["last_access_ts"] = datetime.now(timezone.utc)
                self.cache[object_id] = e
            elif in_cache and cached_version != self.origin_version.get(object_id, 0):
                # cache hit BUT stale (origin was updated after we cached)
                served_latency = float(self.hit_latency_ms)
                served_from_cache = True
                stale = True
                # keep cache entry but note staleness
                e = self.cache.pop(object_id)
                e["last_access_ts"] = datetime.now(timezone.utc)
                self.cache[object_id] = e
            else:
                # miss -> origin
                served_latency = float(origin_latency)
                served_from_cache = False
                stale = False
                # note: TTL baseline may decide to upsert on write; we only upsert if policy instructs (trainer/policy will do)
        # reward calculation (simple)
        latency_saved = (origin_latency - served_latency) / 100.0
        cache_cost = (size_bytes / float(self.max_bytes)) * 0.5 if action == 1 else 0.0
        fill_penalty = (self.bytes_used / float(self.max_bytes)) * 0.2
        reward = latency_saved - cache_cost - fill_penalty

        info = {
            "object_id": object_id,
            "served_latency_ms": served_latency,
            "served_from_cache": served_from_cache,
            "stale": stale,
            "origin_latency_ms": origin_latency,
            "size_bytes": int(size_bytes),
            "was_write": was_write
        }

        # move pointer
        self.index += 1
        done = self.index >= len(self.df)

        next_obs, _ = (None, None)
        if not done:
            next_obs, _ = self._make_obs_for_row(self.df.iloc[self.index])

        return next_obs, float(reward), done, info
