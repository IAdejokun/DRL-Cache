# backend/main.py - the FastAPI app: startup config, CORS, and the endpoints.
import os, datetime
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, func, case
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from db import Base, engine, get_db
import models
from schemas import RequestIn, OutcomeOut, StatsOut

load_dotenv()  # reads .env
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

# --- Baseline in-memory TTL helper using DB state ---

def is_cache_hit(db: Session, object_id: str) -> bool:
    item = db.query(models.CacheItem).filter(models.CacheItem.object_id == object_id).first()
    if not item:
        return False
    age = datetime.datetime.now(datetime.timezone.utc) - item.last_updated_ts
    return age.total_seconds() <= item.ttl_s

def upsert_cache_item(db: Session, object_id: str, size_bytes: int, ttl_s: int = 300):
    now = datetime.datetime.now(datetime.timezone.utc)
    item = db.query(models.CacheItem).filter(models.CacheItem.object_id == object_id).first()
    if item:
        item.size_bytes = size_bytes
        item.last_updated_ts = now
        item.ttl_s = ttl_s
    else:
        item = models.CacheItem(
            object_id=object_id,
            size_bytes=size_bytes,
            last_updated_ts=now,
            ttl_s=ttl_s,
        )
        db.add(item)

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

    # 2) Decide cache vs origin (simple TTL); writes always refresh
    hit = False
    served_latency = req.origin_latency_ms
    staleness = 0

    if req.was_write:
        # refresh/put into cache
        upsert_cache_item(db, req.object_id, req.object_size_bytes)
        hit = False
        served_latency = req.origin_latency_ms
    else:
        if is_cache_hit(db, req.object_id):
            hit = True
            served_latency = 20  # pretend cache is ~20ms
        else:
            # miss -> fetch from origin and store
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

@app.get("/api/stats", response_model=StatsOut)
def get_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(models.Outcome.id)).scalar() or 0

    hits = db.query(
        func.sum(case((models.Outcome.cache_hit == True, 1), else_=0))
    ).scalar() or 0

    stale = db.query(
        func.sum(case((models.Outcome.staleness_s > 0, 1), else_=0))
    ).scalar() or 0

    avg_latency = db.query(func.avg(models.Outcome.served_latency_ms)).scalar() or 0.0

    return StatsOut(
        hit_ratio_pct=round(100.0 * (hits / total), 2) if total else 0.0,
        avg_latency_ms=round(float(avg_latency), 2),
        staleness_pct=round(100.0 * (stale / total), 2) if total else 0.0,
    )