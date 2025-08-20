# backend/main.py - the FastAPI app: startup config, CORS, and the endpoints.
import os, datetime
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, func, case
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from pathlib import Path


from db import Base, engine, get_db
import models
from schemas import RequestIn, OutcomeOut, StatsOut

#load_dotenv()  # reads .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")  # use .env in the same directory as this file

# --- Settings (read once) ---
DEFAULT_TTL = int(os.getenv("CACHE_DEFAULT_TTL_S", "300"))
MAX_BYTES   = int(os.getenv("CACHE_MAX_BYTES", "50000000"))
HIT_MS      = int(os.getenv("CACHE_HIT_LATENCY_MS", "20"))

app = FastAPI(title = "DRL Cache Gateway")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables if they don't exist
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"db": "ok"}

# time & capacity utilities

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)

def total_cache_bytes(db: Session) -> int:
    return int(db.query(func.coalesce(func.sum(models.CacheItem.size_bytes), 0)).scalar() or 0)

def evict_until_fit(db: Session, size_needed: int):
    """
    Evict least-recently-used items until there is room for 'size_needed'
    based on MAX_BYTES.
    """
    used = total_cache_bytes(db)
    if used + size_needed <= MAX_BYTES:
        return  # enough space

    to_free = (used + size_needed) - MAX_BYTES
    freed = 0

    # Evict by oldest last_access_ts first (NULLs first to be safe)
    victims = (
        db.query(models.CacheItem)
          .order_by(models.CacheItem.last_access_ts.asc().nullsfirst())
          .all()
    )
    for v in victims:
        freed += int(v.size_bytes or 0)
        db.delete(v)
        if freed >= to_free:
            break


# --- Baseline in-memory TTL helper using DB state ---

def is_cache_hit(db: Session, object_id: str) -> bool:
    item = (
        db.query(models.CacheItem)
          .filter(models.CacheItem.object_id == object_id)
          .first()
    )
    if not item:
        return False
    age = (now_utc() - item.last_updated_ts).total_seconds()
    hit = age <= item.ttl_s
    if hit:
        # Touch for LRU
        item.last_access_ts = now_utc()
        db.flush()
    return hit

def upsert_cache_item(db: Session, object_id: str, size_bytes: int, ttl_s: int = None):
    ttl = ttl_s if ttl_s is not None else DEFAULT_TTL
    current = (
        db.query(models.CacheItem)
          .filter(models.CacheItem.object_id == object_id)
          .first()
    )
    if current:
        current.size_bytes = size_bytes
        current.last_updated_ts = now_utc()
        current.last_access_ts = now_utc()
        current.ttl_s = ttl
    else:
        # Ensure capacity before inserting a new object
        evict_until_fit(db, size_bytes)
        db.add(models.CacheItem(
            object_id=object_id,
            size_bytes=size_bytes,
            last_updated_ts=now_utc(),
            last_access_ts=now_utc(),
            ttl_s=ttl,
        ))
    db.flush()

@app.post("/api/request", response_model=OutcomeOut)
def handle_request(req: RequestIn, db: Session = Depends(get_db)):
    # 1) Insert request
    q = models.Request(
        ts=datetime.datetime.fromisoformat(req.ts.replace("Z","+00:00")),
        client_id=req.client_id,
        object_id=req.object_id,
        object_size_bytes=req.object_size_bytes,
        origin_latency_ms=req.origin_latency_ms,
        was_write=req.was_write,
    )
    db.add(q)
    db.flush()  # get q.id

    # 2) Decide cache vs origin (TTL + LRU + capacity); writes always refresh
    hit = False
    served_latency = req.origin_latency_ms
    staleness = 0

    if req.was_write:
        # Refresh the cache on writes
        upsert_cache_item(db, req.object_id, req.object_size_bytes)
        hit = False
        served_latency = req.origin_latency_ms
    else:
        if is_cache_hit(db, req.object_id):
            hit = True
            served_latency = HIT_MS  # fast path
        else:
            # Miss -> fetch from origin, make room, then store
            evict_until_fit(db, req.object_size_bytes)
            upsert_cache_item(db, req.object_id, req.object_size_bytes)
            hit = False
            served_latency = req.origin_latency_ms


    # 3) Record outcome
    outcome = models.Outcome(
        request_id=q.id,
        cache_hit=hit,
        served_latency_ms=served_latency,
        staleness_s=staleness,
    )
    db.add(outcome)
    db.commit()

    return OutcomeOut(
        request_id=q.id,
        cache_hit=hit,
        served_latency_ms=served_latency,
        staleness_s=staleness,
    )

@app.get("/api/cache")
def get_cache(db: Session = Depends(get_db)):
    rows = db.query(models.CacheItem).all()
    return [
        {
            "object_id": r.object_id,
            "size_bytes": r.size_bytes,
            "last_updated_ts": r.last_updated_ts.isoformat(),
            "ttl_s": r.ttl_s,
        } for r in rows
    ]

#Add a tiny cache-stats endpoint

@app.get("/api/cache/stats")
def cache_stats(db: Session = Depends(get_db)):
    count = db.query(func.count(models.CacheItem.id)).scalar() or 0
    used = total_cache_bytes(db)
    return {
        "items": int(count),
        "bytes_used": int(used),
        "max_bytes": int(MAX_BYTES),
        "pct_full": round(100.0 * (used / MAX_BYTES), 2) if MAX_BYTES else 0.0
    }

@app.get("/api/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_db)):
    hit_avg = db.query(func.avg(case((models.Outcome.cache_hit == True, 1), else_=0))).scalar() or 0.0
    avg_latency = db.query(func.avg(models.Outcome.served_latency_ms)).scalar() or 0.0
    stale_avg = db.query(func.avg(case((models.Outcome.staleness_s > 0, 1), else_=0))).scalar() or 0.0
    return StatsOut(
        hit_ratio_pct=round(100.0 * float(hit_avg), 2),
        avg_latency_ms=round(float(avg_latency), 2),
        staleness_pct=round(100.0 * float(stale_avg), 2),
    )