# backend/schemas.py - shapes of data going in/out of your API (validation layer)
from pydantic import BaseModel
from typing import Optional

class RequestIn(BaseModel):
    ts: str                 # ISO string
    client_id: Optional[str] = None
    object_id: str
    object_size_bytes: int
    origin_latency_ms: int
    was_write: bool = False

class OutcomeOut(BaseModel):
    request_id: int
    cache_hit: bool
    served_latency_ms: int
    staleness_s: int

class StatsOut(BaseModel):
    hit_ratio_pct: float
    avg_latency_ms: float
    staleness_pct: float
