# rl_agent/env.py
import random
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

class CacheEnv:
    """
    Simple Gym-like environment for caching decisions.
    Observations (float vector):
      [ key_id_norm, in_cache (0/1), age_norm, cache_fill_frac, item_size_norm ]
    Actions:
      0 -> do NOT cache (or do nothing)
      1 -> cache/refresh this item
    Reward:
      positive for latency savings; negative small cost for using space.
    """

    def __init__(
        self,
        num_keys: int = 50,
        max_bytes: int = 50_000_000,
        hit_latency_ms: int = 20,
        miss_latency_range=(150, 350),
        size_kb_range=(5, 500),
        ttl_s: int = 300,
        workload: str = "zipf",  # zipf|uniform|flash (zipf by default)
        zipf_s: float = 1.07,
        seed: int | None = None,
    ):
        self.num_keys = num_keys
        self.max_bytes = max_bytes
        self.hit_latency_ms = hit_latency_ms
        self.miss_latency_range = miss_latency_range
        self.size_kb_range = size_kb_range
        self.ttl_s = ttl_s
        self.workload = workload
        self.zipf_s = zipf_s
        self.rng = random.Random(seed)

        # simple in-memory cache: OrderedDict for LRU semantics
        # map key -> (inserted_ts (float seconds), size_bytes)
        self.cache = OrderedDict()
        self.bytes_used = 0.0
        self.now = 0.0  # simulated time in seconds

        # build zipf weights if needed
        if self.workload == "zipf":
            ranks = np.arange(1, self.num_keys + 1)
            weights = 1.0 / (ranks ** self.zipf_s)
            self.zipf_weights = weights / weights.sum()
        else:
            self.zipf_weights = None

    def _sample_key(self):
        if self.workload == "zipf":
            return int(np.random.choice(np.arange(1, self.num_keys + 1), p=self.zipf_weights))
        else:
            return self.rng.randint(1, self.num_keys)

    def reset(self):
        self.cache.clear()
        self.bytes_used = 0.0
        self.now = 0.0
        # return a first observation
        key = self._sample_key()
        obs, meta = self._make_obs_for_key(key)
        return np.array(obs, dtype=np.float32)

    def _evict_until_fit(self, size_needed: int):
        """Evict LRU items until there's room for size_needed bytes."""
        while self.bytes_used + size_needed > self.max_bytes and len(self.cache) > 0:
            k, (ts, sz) = self.cache.popitem(last=False)  # pop oldest
            self.bytes_used -= sz

    def _make_obs_for_key(self, key: int):
        """
        Build observation for a given key.
        Returns (obs_vector, meta) where meta contains item_size_bytes and whether hit.
        """
        # normalize key id
        key_norm = key / float(self.num_keys)
        in_cache = 0
        age_norm = 1.0  # 1.0 = large/expired or unknown
        item_size = self.rng.randint(self.size_kb_range[0] * 1024, self.size_kb_range[1] * 1024)

        if key in self.cache:
            inserted_ts, size_bytes = self.cache[key]
            age = self.now - inserted_ts
            if age <= self.ttl_s:
                in_cache = 1
                age_norm = min(age / float(self.ttl_s), 1.0)
            else:
                # expired — treat as miss and remove
                try:
                    del self.cache[key]
                    self.bytes_used -= size_bytes
                except KeyError:
                    pass
                in_cache = 0
                age_norm = 1.0

        cache_fill_frac = min(max(self.bytes_used / float(self.max_bytes), 0.0), 1.0)
        obs = [key_norm, float(in_cache), age_norm, cache_fill_frac, item_size / float(self.max_bytes)]
        meta = {"key": key, "item_size": item_size, "in_cache": bool(in_cache)}
        return obs, meta

    def step(self, action: int):
        """
        Execute one request with agent action for caching decision.
        Returns: next_state, reward, done, info
        """
        # advance simulated time a little
        self.now += 1.0  # 1 second per request (simplification)

        key = self._sample_key()
        obs, meta = self._make_obs_for_key(key)
        in_cache = meta["in_cache"]
        item_size = meta["item_size"]

        # determine origin latency (miss) and served latency
        origin_latency = self.rng.randint(self.miss_latency_range[0], self.miss_latency_range[1])
        if in_cache:
            # hit
            served_latency = self.hit_latency_ms
            hit = True
        else:
            served_latency = origin_latency
            hit = False

        # Apply action: 0 = do nothing, 1 = ensure item is cached/refresh
        cached_now = False
        if action == 1:
            # try to cache (refresh or insert)
            if not in_cache:
                # evict until fit
                self._evict_until_fit(item_size)
                # insert
                self.cache[key] = (self.now, item_size)
                # move to end -> most recently used
                try:
                    self.cache.move_to_end(key, last=True)
                except Exception:
                    pass
                self.bytes_used += item_size
                cached_now = True
            else:
                # refresh timestamp (touch)
                inserted_ts, sz = self.cache.pop(key)
                self.cache[key] = (self.now, sz)
                cached_now = True

        # compute reward:
        # Reward is proportional to latency saved on this step:
        latency_saved = (origin_latency - served_latency) / 100.0  # scaled
        # caching cost penalty (size)
        cache_cost = (item_size / float(self.max_bytes)) * 0.5 if cached_now else 0.0
        # small penalty for high fill fraction to encourage space efficiency
        fill_penalty = (self.bytes_used / float(self.max_bytes)) * 0.2
        reward = latency_saved - cache_cost - fill_penalty

        # next observation
        next_obs, _ = self._make_obs_for_key(key)

        # done is always False in this continuing task (we do episodes in train loop)
        done = False
        info = {"origin_latency": origin_latency, "served_latency": served_latency, "hit": hit, "cached_now": cached_now}
        return np.array(next_obs, dtype=np.float32), float(reward), done, info
