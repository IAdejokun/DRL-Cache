# backend/main.py - the FastAPI app: startup config, CORS, and the endpoints.
import os, datetime
from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, func, case
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from pathlib import Path
import subprocess, sys, threading, time
from typing import Optional

from db import Base, engine, get_db, SessionLocal
import models
from schemas import RequestIn, OutcomeOut, StatsOut


#load_dotenv()  # reads .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")  # use .env in the same directory as this file

SAMPLE_HZ = 1  # once per second

SIM_DIR = os.getenv("SIM_DIR", "../simulator")
PY_EXE = os.getenv("PY_EXE", "python")


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

# @app.on_event("startup")
# def on_startup():
#     Base.metadata.create_all(bind=engine)

def sample_stats_loop():
    with SessionLocal() as db:  # reuse your SessionLocal from db.py
        while True:
            # read current stats using same logic as /api/stats
            hit_avg = db.query(func.avg(case((models.Outcome.cache_hit == True, 1), else_=0))).scalar() or 0.0
            avg_latency = db.query(func.avg(models.Outcome.served_latency_ms)).scalar() or 0.0
            stale_avg = db.query(func.avg(case((models.Outcome.staleness_s > 0, 1), else_=0))).scalar() or 0.0

            snap = models.StatSnapshot(
                run_id=None,  # we’ll attach a run_id when a run is active (optional)
                hit_ratio_pct=round(100.0*float(hit_avg),2),
                avg_latency_ms=round(float(avg_latency),2),
                staleness_pct=round(100.0*float(stale_avg),2),
            )
            db.add(snap)
            db.commit()
            time.sleep(1.0 / SAMPLE_HZ)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    threading.Thread(target=sample_stats_loop, daemon=True).start()


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

@app.get("/api/history")
def get_history(window: int = Query(120, ge=10, le=3600), run_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(models.StatSnapshot).order_by(models.StatSnapshot.ts.desc())
    if run_id is not None:
        q = q.filter(models.StatSnapshot.run_id == run_id)
    rows = q.limit(window).all()
    # reverse chronological → chronological
    rows = list(reversed(rows))
    return [
        {
            "ts": r.ts.isoformat(),
            "hit_ratio_pct": float(r.hit_ratio_pct or 0),
            "avg_latency_ms": float(r.avg_latency_ms or 0),
            "staleness_pct": float(r.staleness_pct or 0),
        } for r in rows
    ]

@app.get("/api/runs")
def list_runs(db: Session = Depends(get_db)):
    rows = db.query(models.Run).order_by(models.Run.id.desc()).limit(20).all()
    return [
        { "id": r.id, "started_ts": r.started_ts.isoformat(),
          "workload": r.workload, "minutes": r.minutes, "rps": r.rps,
          "rate": r.rate, "status": r.status }
        for r in rows
    ]

@app.post("/api/experiments/run")
def run_experiment(payload: dict, db: Session = Depends(get_db)):
    workload = payload.get("workload", "zipf")          # zipf|flash|writeheavy
    minutes = int(payload.get("minutes", 2))
    rps     = int(payload.get("rps", 5))
    rate    = int(payload.get("rate", 10))

    # Create a run row
    run = models.Run(workload=workload, minutes=minutes, rps=rps, rate=rate, status="running")
    db.add(run); db.commit(); db.refresh(run)

    # Build CSV path and command
    csv_path = os.path.abspath(os.path.join(SIM_DIR, f"data/run_{run.id}.csv"))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    make_cmd = [PY_EXE, os.path.join(SIM_DIR, "make_csv.py"),
                "--workload", workload, "--minutes", str(minutes),
                "--rps", str(rps), "--objects", "200", "--clients", "50",
                "--outfile", csv_path]

    replay_cmd = [PY_EXE, os.path.join(SIM_DIR, "replay.py"),
                  "--file", csv_path, "--base", "http://127.0.0.1:8000",
                  "--rate", str(rate)]

    def worker():
        try:
            subprocess.check_call(make_cmd)
            subprocess.check_call(replay_cmd)
            run.status = "done"
        except Exception as e:
            run.status = "error"
        finally:
            with SessionLocal() as s:
                s.merge(run); s.commit()

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run.id, "status": "started"}
