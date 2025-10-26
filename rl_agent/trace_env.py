# rl_agent/trace_env.py
import pandas as pd
import numpy as np
import random
import os

class TraceEnv:
    """
    Environment wrapper for CSV traces.
    Supports alias fallback:
        - ts (aliases: timestamp, time, datetime, request_ts)
        - key (aliases: cache_key, object_id, id, item)
        - size (aliases: size, bytes, content_length, object_size, object_size_bytes)
        - was_write (aliases: was_write, write_flag, is_write, write, op, operation, op_type)
    """

    def __init__(self, csv_path, max_bytes=50_000_000, ttl_s=300, hit_latency_ms=20, seed=42):
        self.csv_path = csv_path
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self.hit_latency_ms = hit_latency_ms

        random.seed(seed)
        np.random.seed(seed)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Trace file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Expected columns + aliases mapping
        required_mapping = {
            "ts": ["ts", "timestamp", "time", "datetime", "request_ts"],
            "key": ["key", "cache_key", "object_id", "id", "item"],
            "size": ["size", "bytes", "content_length", "object_size", "object_size_bytes"],
            "was_write": ["was_write", "write_flag", "is_write", "write", "op", "operation", "op_type"]
        }

        renamed_columns = {}
        for canon, aliases in required_mapping.items():
            if canon not in self.df.columns:
                found = None
                for alias in aliases:
                    if alias in self.df.columns:
                        found = alias
                        break
                if found:
                    renamed_columns[found] = canon
                else:
                    raise ValueError(f"Trace missing required column or alias for: {canon} (aliases tried: {aliases})")

        if renamed_columns:
            print(f"[TraceEnv] Auto-alias mapping applied: {renamed_columns}")
            self.df.rename(columns=renamed_columns, inplace=True)

        # ✅ Normalize timestamp to numeric values (seconds since first event)
        if not np.issubdtype(self.df["ts"].dtype, np.number):
            try:
                start_ts = pd.to_datetime(self.df["ts"].iloc[0])
                self.df["ts"] = (pd.to_datetime(self.df["ts"]) - start_ts).dt.total_seconds()
                print("[TraceEnv] Converted timestamp column to numeric seconds")
            except Exception:
                raise ValueError("Column 'ts' could not be converted to numeric timestamps")

        # ✅ Normalize size to numeric just in case
        self.df["size"] = pd.to_numeric(self.df["size"], errors="coerce").fillna(0)

        # ✅ Normalize was_write to 0/1
        if self.df["was_write"].dtype == object:
            self.df["was_write"] = self.df["was_write"].apply(
                lambda x: 1 if str(x).lower() in ["1", "true", "w", "write"] else 0
            )

        # Sort by timestamp
        self.df.sort_values("ts", inplace=True, ascending=True)
        self.df.reset_index(drop=True, inplace=True)

        self.cache = {}  # key -> (expiry_time, size)
        self.current_bytes = 0
        self.index = 0
        self.total_requests = len(self.df)

    def reset(self):
        self.cache.clear()
        self.current_bytes = 0
        self.index = 0
        if self.total_requests == 0:
            raise ValueError("CSV loaded but empty.")
        return self._get_obs()

    def _get_obs(self):
        if self.index >= self.total_requests:
            return None, {}
        row = self.df.iloc[self.index]
        key = row["key"]
        in_cache = 1 if key in self.cache and self.cache[key][0] > row["ts"] else 0
        remaining_ttl = 0
        if key in self.cache:
            expiry = self.cache[key][0]
            remaining_ttl = max(0, expiry - row["ts"])

        obs = np.array([
            in_cache,
            self.current_bytes,
            row["size"],
            remaining_ttl,
            row["was_write"]
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        if self.index >= self.total_requests:
            return None, 0.0, True, {}

        row = self.df.iloc[self.index]
        key, size, ts, is_write = row["key"], row["size"], row["ts"], row["was_write"]

        served_from_cache = False
        stale = False

        if key in self.cache:
            expiry, obj_size = self.cache[key]
            if expiry >= ts:
                served_from_cache = True
            else:
                served_from_cache = True
                stale = True

        if action == 1 or is_write == 1:
            while self.current_bytes + size > self.max_bytes and self.cache:
                evict_key = next(iter(self.cache))
                _, evict_size = self.cache.pop(evict_key)
                self.current_bytes -= evict_size

            expiry_ts = ts + self.ttl_s
            if key not in self.cache:
                self.current_bytes += size
            self.cache[key] = (expiry_ts, size)
            served_from_cache = True
            stale = False

        if served_from_cache and not stale:
            latency = self.hit_latency_ms
        else:
            latency = self.hit_latency_ms * 5

        reward = 1 if served_from_cache and not stale else -1

        info = {
            "served_latency_ms": latency,
            "served_from_cache": served_from_cache,
            "stale": stale
        }

        self.index += 1
        done = self.index >= self.total_requests

        return self._get_obs()[0] if not done else None, reward, done, info
