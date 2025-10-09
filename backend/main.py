# backend/main.py - the FastAPI app: startup config, CORS, and the endpoints.
# top-of-file imports (replace your existing import block)
import os
import sys
import random
import time
import subprocess
import threading
import json
import glob
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, func, case
from sqlalchemy.orm import Session

from db import Base, engine, get_db, SessionLocal
import models
from schemas import RequestIn, OutcomeOut, StatsOut

# datetime imports — use timezone-aware datetimes consistently
from datetime import datetime, timezone, timedelta

# load .env placed next to backend/main.py
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ensure repo root is on sys.path so `rl_agent` can be imported when running from backend/
REPO_ROOT = Path(__file__).resolve().parents[1]  # ../ from backend/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- DRL Agent loading ---
AGENT = None
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "rl_agent", "model.pt"))

try:
    import torch
    # ensure rl_agent package is importable; path assumed relative to repo root
    from rl_agent.dqn import DQNAgent
    # state_dim must match env (5)
    if os.path.exists(MODEL_PATH):
        AGENT = DQNAgent.load(MODEL_PATH, state_dim=5)
        print("Loaded DRL agent from", MODEL_PATH)
    else:
        print("DRL model not found at", MODEL_PATH, "; DRL mode will fallback to TTL.")
except Exception as e:
    AGENT = None
    print("DRL agent not available:", repr(e))

SAMPLE_HZ = 1  # once per second

SIM_DIR = os.getenv("SIM_DIR", "../simulator")
PY_EXE = os.getenv("PY_EXE", sys.executable or "python")

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

# --- Simple in-memory cache simulation for testing ---
# in-memory cache_store: key -> (inserted_ts: datetime, value, size_bytes:int)
cache_store = {}
# Use DEFAULT_TTL (env) for simulation TTL by default
CACHE_TTL = DEFAULT_TTL

def cache_bytes_used():
    return sum(int(v[2]) for v in cache_store.values())

def _make_object_id(n: int) -> str:
    return f"item{n}"

def predict_drl_action(obs):
    """
    obs: list/np array length=5 (same as train env observation)
    returns: integer action 0/1
    """
    if AGENT is None:
        return 0  # fallback: do nothing cache-wise (acts like TTL fallback)
    # no exploration at inference: epsilon=0.0
    try:
        return AGENT.act(obs, epsilon=0.0)
    except Exception as e:
        print("Error during agent act:", e)
        return 0

def simulate_request(mode: str = "ttl"):
    """
    In-memory simulation for /api/simulate.
    Returns:
      object_id (str), hit (bool), served_latency_ms (float), staleness_s (float), size_bytes (int)
    """
    # choose an object
    key_num = random.randint(1, 10)
    object_id = _make_object_id(key_num)
    now = now_utc()

    # check in-memory cache
    entry = cache_store.get(object_id)
    in_cache = False
    age_s = None
    size_bytes = random.randint(5 * 1024, 500 * 1024)  # default simulated size

    if entry:
        inserted_ts, value, stored_size = entry
        age_s = (now - inserted_ts).total_seconds()
        size_bytes = stored_size or size_bytes
        in_cache = (age_s <= CACHE_TTL)

    # Mode: TTL
    if mode == "ttl":
        if in_cache:
            # hit
            served_latency = float(HIT_MS)
            staleness = max(0.0, age_s) if age_s is not None else 0.0
            # touch for LRU semantics in in-memory cache
            cache_store[object_id] = (now, value, size_bytes)
            return object_id, True, served_latency, staleness, size_bytes
        else:
            # miss -> fetch origin, insert into cache
            origin_latency = random.uniform(100, 200)
            cache_store[object_id] = (now, random.random(), size_bytes)
            return object_id, False, float(origin_latency), 0.0, size_bytes

    # Mode: DRL (use AGENT on same in-memory state)
    if mode == "drl":
        # build observation [key_norm, in_cache, age_norm, cache_fill_frac, size_norm]
        key_norm = key_num / 10.0
        in_cache_f = 1.0 if in_cache else 0.0
        age_norm = min((age_s / float(CACHE_TTL)) if age_s is not None else 1.0, 1.0)
        fill_frac = min(cache_bytes_used() / float(MAX_BYTES), 1.0) if MAX_BYTES else 0.0
        size_norm = size_bytes / float(MAX_BYTES) if MAX_BYTES else 0.0
        obs = [key_norm, float(in_cache_f), float(age_norm), float(fill_frac), float(size_norm)]

        action = predict_drl_action(obs)  # 0 or 1
        if action == 1:
            # cache/refresh
            cache_store[object_id] = (now, random.random(), size_bytes)
            served_latency = float(HIT_MS)
            staleness = 0.0 if not in_cache else (age_s or 0.0)
            return object_id, True, served_latency, staleness, size_bytes
        else:
            # do not cache or refresh: serve from cache if still valid, else origin
            if in_cache:
                served_latency = float(HIT_MS)
                staleness = max(0.0, age_s or 0.0)
                # keep the entry but update LRU timestamp
                cache_store[object_id] = (now, entry[1], size_bytes)
                return object_id, True, served_latency, staleness, size_bytes
            else:
                origin_latency = random.uniform(100, 200)
                # no caching on this action
                return object_id, False, float(origin_latency), 0.0, size_bytes

    # Mode: HYBRID (simple mix)
    if mode == "hybrid":
        # use DRL action if agent available, otherwise TTL
        if AGENT:
            return simulate_request("drl")
        else:
            return simulate_request("ttl")

    # default fallback: TTL-like
    return simulate_request("ttl")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"db": "ok"}

# time & capacity utilities

def now_utc():
    return datetime.now(timezone.utc)

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
        ts=datetime.fromisoformat(req.ts.replace("Z","+00:00")),
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

# --------------------- Day 10: model registry & training/eval endpoints ---------------------

# Setup RL agent models directory & registry import (file-based registry living in rl_agent/models/)
RL_AGENT_DIR = REPO_ROOT / "rl_agent"
RL_MODELS_DIR = RL_AGENT_DIR / "models"
RL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Try to import registry helpers; provide safe fallback if not available.
try:
    from rl_agent.registry import list_models, add_model_entry, find_model_by_id
except Exception as e:
    print("Could not import rl_agent.registry:", e)
    # fallback in-memory listing (not persisted) — small shim
    _REGISTRY_FALLBACK = []
    def list_models():
        return _REGISTRY_FALLBACK
    def add_model_entry(p, meta=None):
        entry = {"id": datetime.utcnow().strftime("%Y%m%d%H%M%S"), "path": str(p), "created_ts": datetime.utcnow().isoformat() + "Z", "meta": meta or {}}
        _REGISTRY_FALLBACK.append(entry)
        return entry
    def find_model_by_id(mid):
        for m in _REGISTRY_FALLBACK:
            if m["id"] == mid:
                return m
        return None

def _train_background(trace: str, epochs: int, out_name: str):
    """
    Runs a training subprocess and registers the resulting model on success.
    out_name should be a filename under rl_agent/models (e.g. model_20251006.pt)
    """
    out_path = str((RL_MODELS_DIR / out_name).resolve())
    cmd = [PY_EXE, str(RL_AGENT_DIR / "train_offline.py"),
           "--trace", trace, "--epochs", str(epochs), "--out", out_path]
    try:
        print("Starting training subprocess:", " ".join(cmd))
        subprocess.check_call(cmd)
        # register model
        entry = add_model_entry(out_path, meta={"trace": trace, "epochs": epochs})
        print("Training finished, registered model:", entry)
    except Exception as e:
        print("Training failed:", repr(e))

@app.post("/api/models/train")
def api_train_model(payload: dict, background: BackgroundTasks = None):
    """
    Starts a background training job.
    payload: { "trace": "simulator/data/run_1.csv", "epochs": 20 }
    Returns a job handle (best-effort).
    """
    trace = payload.get("trace", str(REPO_ROOT / "simulator" / "data" / "run_1.csv"))
    epochs = int(payload.get("epochs", 20))
    # generate deterministic name
    out_name = f"model_{int(time.time())}.pt"

    # Basic validation
    if not Path(trace).exists():
        raise HTTPException(status_code=400, detail=f"Trace file not found: {trace}")

    # fire-and-forget: use BackgroundTasks if available, else thread
    if background is not None:
        background.add_task(_train_background, trace, epochs, out_name)
    else:
        threading.Thread(target=_train_background, args=(trace, epochs, out_name), daemon=True).start()

    return {"status": "started", "trace": trace, "epochs": epochs, "out_name": out_name}

@app.get("/api/models")
def api_list_models():
    return list_models()

@app.post("/api/models/evaluate")
def api_evaluate_model(payload: dict):
    """
    Evaluate a model on a given trace (synchronous).
    payload: { "model_id": "<id>" , "trace": "simulator/data/run_1.csv", "out": "rl_agent/logs/eval_XXX.csv", "plot": false }
    """
    model_id = payload.get("model_id")
    trace = payload.get("trace", str(REPO_ROOT / "simulator" / "data" / "run_1.csv"))
    out = payload.get("out", str(REPO_ROOT / "rl_agent" / "logs" / f"eval_{int(time.time())}.csv"))
    plot = bool(payload.get("plot", False))

    model_entry = None
    model_path = None
    if model_id:
        model_entry = find_model_by_id(model_id)
        if not model_entry:
            raise HTTPException(status_code=404, detail="Model id not found")
        model_path = model_entry["path"]
    else:
        model_path = payload.get("model_path")  # allow direct path
        if model_path and not Path(model_path).exists():
            raise HTTPException(status_code=400, detail=f"model_path not found: {model_path}")
        if not model_path:
            raise HTTPException(status_code=400, detail="model_id or model_path required")

    # use evaluation helpers from rl_agent.evaluate
    try:
        from rl_agent.evaluate import evaluate_ttl_from_trace, evaluate_drl_from_trace, save_report_csv, plot_comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluate module import failed: {e}")

    ttl_metrics = evaluate_ttl_from_trace(trace)
    drl_metrics = None
    if model_path and Path(model_path).exists():
        drl_metrics = evaluate_drl_from_trace(trace, model_path)
    # Ensure output directory exists
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_report_csv(str(out_path), ttl_metrics, drl_metrics)
    if plot:
        plot_path = str(out_path.with_suffix(".png"))
        plot_comparison(ttl_metrics, drl_metrics, save_path=plot_path)

    return {"out": str(out_path), "ttl": ttl_metrics, "drl": drl_metrics}

# Optional: auto-retrain poller (disabled by default; enable if you want automated retraining)
def auto_retrain_poller(poll_dir: str = str(REPO_ROOT / "simulator" / "data"), interval_s: int = 60):
    seen = set()
    while True:
        files = sorted(glob.glob(os.path.join(poll_dir, "*.csv")))
        for f in files:
            if f not in seen:
                seen.add(f)
                threading.Thread(target=_train_background, args=(f, 5, f"auto_{int(time.time())}.pt"), daemon=True).start()
        time.sleep(interval_s)

# threading.Thread(target=auto_retrain_poller, daemon=True).start()
# ----------------------------------------------------------------------------------------

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

@app.post("/api/simulate")
def run_simulation(
    mode: str = Query("ttl", enum=["ttl", "drl", "hybrid"]),
    db: Session = Depends(get_db)
):
    total_requests = 50
    hits, total_latency, total_staleness = 0, 0.0, 0.0

    for _ in range(total_requests):
        object_id, hit, served_latency, staleness_s, size_bytes = simulate_request(mode)
        total_latency += served_latency
        total_staleness += staleness_s
        if hit:
            hits += 1

        # create a Request row for traceability (similar to handle_request)
        q = models.Request(
            ts=now_utc(),
            client_id="sim",
            object_id=object_id,
            object_size_bytes=size_bytes,
            origin_latency_ms=served_latency if not hit else HIT_MS,
            was_write=False,
        )
        db.add(q)
        db.flush()  # assign q.id

        # Store Outcome using consistent column names
        outcome = models.Outcome(
            request_id=q.id if hasattr(q, "id") else None,
            cache_hit=bool(hit),
            served_latency_ms=float(served_latency),
            staleness_s=float(staleness_s),
        )
        db.add(outcome)
        db.commit()

    avg_latency = total_latency / total_requests
    hit_ratio = (hits / total_requests) * 100
    avg_staleness = (total_staleness / total_requests) * 100

    return {
        "mode": mode,
        "total_requests": total_requests,
        "hit_ratio_pct": round(hit_ratio, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "staleness_pct": round(avg_staleness, 2),
    }
