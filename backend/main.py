# backend/main.py - FastAPI app: startup config, CORS, endpoints
import os
import sys
import random
import time
import subprocess
import threading
import json
import glob
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

from sqlalchemy import text, func, case
from sqlalchemy.orm import Session

from db import Base, engine, get_db, SessionLocal
import models
from schemas import RequestIn, OutcomeOut, StatsOut

from datetime import datetime, timezone

# --- load environment (backend/.env expected) ---
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (../ from backend/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- DRL Agent loading (best-effort) ---
AGENT = None
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "rl_agent", "model.pt")
)

try:
    from rl_agent.dqn import DQNAgent  # may fail if rl_agent not importable
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
print("SIM_DIR resolved to:", os.path.abspath(SIM_DIR))
PY_EXE = os.getenv("PY_EXE", sys.executable or "python")

# --- Settings (read once) ---
DEFAULT_TTL = int(os.getenv("CACHE_DEFAULT_TTL_S", "300"))
MAX_BYTES = int(os.getenv("CACHE_MAX_BYTES", "50000000"))
HIT_MS = int(os.getenv("CACHE_HIT_LATENCY_MS", "20"))

app = FastAPI(title="DRL Cache Gateway")

# Terse error surfaces for client; keep server logs for details
@app.exception_handler(Exception)
async def unhandled_exc_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Server error", "hint": str(exc)[:200]})

@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": "Validation error", "errors": exc.errors()})

# --- CORS (localhost + 127.0.0.1) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup: create tables + sampler thread ---
def sample_stats_loop():
    with SessionLocal() as db:
        while True:
            hit_avg = db.query(func.avg(case((models.Outcome.cache_hit == True, 1), else_=0))).scalar() or 0.0
            avg_latency = db.query(func.avg(models.Outcome.served_latency_ms)).scalar() or 0.0
            stale_avg = db.query(func.avg(case((models.Outcome.staleness_s > 0, 1), else_=0))).scalar() or 0.0
            snap = models.StatSnapshot(
                run_id=None,
                hit_ratio_pct=round(100.0 * float(hit_avg), 2),
                avg_latency_ms=round(float(avg_latency), 2),
                staleness_pct=round(100.0 * float(stale_avg), 2),
            )
            db.add(snap)
            db.commit()
            time.sleep(1.0 / SAMPLE_HZ)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    threading.Thread(target=sample_stats_loop, daemon=True).start()

# --- In-memory cache simulation helpers ---
cache_store: Dict[str, tuple] = {}
CACHE_TTL = DEFAULT_TTL

def cache_bytes_used():
    return sum(int(v[2]) for v in cache_store.values())

def _make_object_id(n: int) -> str:
    return f"item{n}"

def predict_drl_action(obs):
    if AGENT is None:
        return 0
    try:
        return AGENT.act(obs, epsilon=0.0)
    except Exception as e:
        print("Error during agent act:", e)
        return 0

def simulate_request(mode: str = "ttl"):
    key_num = random.randint(1, 10)
    object_id = _make_object_id(key_num)
    now = now_utc()

    entry = cache_store.get(object_id)
    in_cache = False
    age_s = None
    size_bytes = random.randint(5 * 1024, 500 * 1024)

    if entry:
        inserted_ts, value, stored_size = entry
        age_s = (now - inserted_ts).total_seconds()
        size_bytes = stored_size or size_bytes
        in_cache = (age_s <= CACHE_TTL)

    if mode == "ttl":
        if in_cache:
            served_latency = float(HIT_MS)
            staleness = max(0.0, age_s) if age_s is not None else 0.0
            cache_store[object_id] = (now, value, size_bytes)
            return object_id, True, served_latency, staleness, size_bytes
        else:
            origin_latency = random.uniform(100, 200)
            cache_store[object_id] = (now, random.random(), size_bytes)
            return object_id, False, float(origin_latency), 0.0, size_bytes

    if mode == "drl":
        key_norm = key_num / 10.0
        in_cache_f = 1.0 if in_cache else 0.0
        age_norm = min((age_s / float(CACHE_TTL)) if age_s is not None else 1.0, 1.0)
        fill_frac = min(cache_bytes_used() / float(MAX_BYTES), 1.0) if MAX_BYTES else 0.0
        size_norm = size_bytes / float(MAX_BYTES) if MAX_BYTES else 0.0
        obs = [key_norm, float(in_cache_f), float(age_norm), float(fill_frac), float(size_norm)]

        action = predict_drl_action(obs)
        if action == 1:
            cache_store[object_id] = (now, random.random(), size_bytes)
            served_latency = float(HIT_MS)
            staleness = 0.0 if not in_cache else (age_s or 0.0)
            return object_id, True, served_latency, staleness, size_bytes
        else:
            if in_cache:
                served_latency = float(HIT_MS)
                staleness = max(0.0, age_s or 0.0)
                cache_store[object_id] = (now, entry[1], size_bytes)
                return object_id, True, served_latency, staleness, size_bytes
            else:
                origin_latency = random.uniform(100, 200)
                return object_id, False, float(origin_latency), 0.0, size_bytes

    if mode == "hybrid":
        if AGENT:
            return simulate_request("drl")
        else:
            return simulate_request("ttl")

    return simulate_request("ttl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/db")
def health_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"db": "ok"}

# --- time & capacity utilities ---
def now_utc():
    return datetime.now(timezone.utc)

def total_cache_bytes(db: Session) -> int:
    # Force int to avoid Decimal leaking out of SUM()
    val = db.query(func.coalesce(func.sum(models.CacheItem.size_bytes), 0)).scalar()
    try:
        return int(val or 0)
    except Exception:
        return int(float(val or 0))

def evict_until_fit(db: Session, size_needed: int):
    used = total_cache_bytes(db)
    if used + size_needed <= MAX_BYTES:
        return
    to_free = (used + size_needed) - MAX_BYTES
    freed = 0
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

# --- DB cache helpers ---
def is_cache_hit(db: Session, object_id: str) -> bool:
    item = db.query(models.CacheItem).filter(models.CacheItem.object_id == object_id).first()
    if not item:
        return False
    age = (now_utc() - item.last_updated_ts).total_seconds()
    hit = age <= item.ttl_s
    if hit:
        item.last_access_ts = now_utc()
        db.flush()
    return hit

def upsert_cache_item(db: Session, object_id: str, size_bytes: int, ttl_s: int = None):
    ttl = ttl_s if ttl_s is not None else DEFAULT_TTL
    current = db.query(models.CacheItem).filter(models.CacheItem.object_id == object_id).first()
    if current:
        current.size_bytes = size_bytes
        current.last_updated_ts = now_utc()
        current.last_access_ts = now_utc()
        current.ttl_s = ttl
    else:
        evict_until_fit(db, size_bytes)
        db.add(models.CacheItem(
            object_id=object_id,
            size_bytes=size_bytes,
            last_updated_ts=now_utc(),
            last_access_ts=now_utc(),
            ttl_s=ttl,
        ))
    db.flush()

# --- Config probe (optional) ---
REDACT = {"DATABASE_URL", "OPENAI_API_KEY"}

@app.get("/api/config")
def api_config_probe():
    env_pairs: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k in REDACT:
            env_pairs[k] = "***"
        elif k.startswith("CACHE_") or k in {"FRONTEND_ORIGIN", "SIM_DIR", "PY_EXE", "MODEL_PATH"}:
            env_pairs[k] = v
    return {
        "cwd": os.getcwd(),
        "base_dir": str(BASE_DIR),
        "repo_root": str(REPO_ROOT),
        "env": env_pairs,
    }

# --- Request handling (records into DB) ---
@app.post("/api/request", response_model=OutcomeOut)
def handle_request(req: RequestIn, db: Session = Depends(get_db)):
    q = models.Request(
        ts=datetime.fromisoformat(req.ts.replace("Z", "+00:00")),
        client_id=req.client_id,
        object_id=req.object_id,
        object_size_bytes=req.object_size_bytes,
        origin_latency_ms=req.origin_latency_ms,
        was_write=req.was_write,
    )
    db.add(q)
    db.flush()

    hit = False
    served_latency = req.origin_latency_ms
    staleness = 0

    if req.was_write:
        upsert_cache_item(db, req.object_id, req.object_size_bytes)
        hit = False
        served_latency = req.origin_latency_ms
    else:
        if is_cache_hit(db, req.object_id):
            hit = True
            served_latency = HIT_MS
        else:
            evict_until_fit(db, req.object_size_bytes)
            upsert_cache_item(db, req.object_id, req.object_size_bytes)
            hit = False
            served_latency = req.origin_latency_ms

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

# --------------------- Model registry & training/eval ---------------------
RL_AGENT_DIR = REPO_ROOT / "rl_agent"
RL_MODELS_DIR = RL_AGENT_DIR / "models"
RL_LOGS_DIR = RL_AGENT_DIR / "logs"
RL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = RL_MODELS_DIR / "registry.json"
ACTIVE_MODEL_FILE = RL_AGENT_DIR / "model.pt"  # canonical active path

def load_registry() -> List[Dict[str, Any]]:
    try:
        if not REGISTRY_PATH.exists():
            REGISTRY_PATH.write_text("[]")
            return []
        data = json.loads(REGISTRY_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception as e:
        print("Failed to load registry:", e)
        return []

def save_registry(entries: List[Dict[str, Any]]):
    try:
        tmp = REGISTRY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(REGISTRY_PATH)
    except Exception as e:
        print("Failed to save registry:", e)

def add_model_entry(path: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entries = load_registry()
    ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    entry = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "path": str(path),
        "created_ts": ts,
        "meta": meta or {},
    }
    entries.append(entry)
    save_registry(entries)
    return entry

def list_models() -> List[Dict[str, Any]]:
    return load_registry()

def find_model_by_id(mid: str) -> Optional[Dict[str, Any]]:
    for m in load_registry():
        if str(m.get("id")) == str(mid):
            return m
    return None

def get_active_model_path() -> Optional[str]:
    if ACTIVE_MODEL_FILE.exists():
        return str(ACTIVE_MODEL_FILE.resolve())
    return None

def load_agent_if_possible(model_path: str) -> bool:
    global AGENT, MODEL_PATH
    try:
        from rl_agent.dqn import DQNAgent
        AGENT = DQNAgent.load(model_path, state_dim=5)
        MODEL_PATH = model_path
        print(f"[MODEL] Active agent loaded from {model_path}")
        return True
    except Exception as e:
        print(f"[MODEL] Failed to load agent: {e}")
        return False

def _train_background(trace: str, epochs: int, out_name: str):
    out_path = str((RL_MODELS_DIR / out_name).resolve())
    cmd = [PY_EXE, "-m", "rl_agent.train_offline", "--trace", trace, "--epochs", str(epochs), "--out", out_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    try:
        print("Starting training subprocess:", " ".join(cmd))
        subprocess.check_call(cmd, env=env)
        entry = add_model_entry(out_path, meta={"trace": trace, "epochs": epochs})
        print("Training finished, registered model:", entry)
    except Exception as e:
        print("Training failed:", repr(e))

@app.get("/api/models/active")
def api_models_active():
    p = get_active_model_path()
    return {"path": p, "loaded": AGENT is not None}

@app.post("/api/models/promote")
def api_models_promote(payload: dict):
    """
    Promote a model to 'active':
      - payload: { model_id?: str, model_path?: str, path?: str }
      - copies the chosen file to rl_agent/model.pt
      - attempts to hot-reload the in-memory agent
    """
    model_id = payload.get("model_id")
    model_path = payload.get("model_path") or payload.get("path")  # accept 'path' alias

    chosen_path = None
    if model_id:
        chosen_entry = find_model_by_id(str(model_id))
        if not chosen_entry:
            raise HTTPException(status_code=404, detail="Model id not found")
        chosen_path = chosen_entry.get("path")
    elif model_path:
        chosen_path = model_path

    if not chosen_path:
        raise HTTPException(status_code=400, detail="model_id or model_path required")

    src = Path(chosen_path)
    if not src.exists():
        raise HTTPException(status_code=400, detail=f"Model file not found: {src}")

    ACTIVE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = src.read_bytes()
        ACTIVE_MODEL_FILE.write_bytes(data)  # Windows-friendly copy
        print(f"[MODEL] Promoted {src} -> {ACTIVE_MODEL_FILE}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy model: {e}")

    ok = load_agent_if_possible(str(ACTIVE_MODEL_FILE))
    return {"status": "ok", "active": str(ACTIVE_MODEL_FILE), "loaded": ok}

@app.post("/api/models/registry/reset")
def api_models_registry_reset():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text("[]", encoding="utf-8")
    return {"status": "ok", "message": "registry cleared"}

@app.post("/api/models/train")
def api_train_model(payload: dict, background: BackgroundTasks = None):
    trace = payload.get("trace", "simulator/data/run_1.csv")

    # Resolve trace path: try provided; repo relative; backend relative
    trace_path = None
    for p in [Path(trace), REPO_ROOT / trace, Path(__file__).resolve().parents[1] / trace]:
        if p.exists():
            trace_path = p
            break
    if trace_path is None:
        raise HTTPException(status_code=400, detail=f"Trace file not found (tried repo/base): {trace}")

    epochs = int(payload.get("epochs", 20))
    out_name = f"model_{int(time.time())}.pt"
    trace_arg = str(Path(trace_path))
    print("TRACE RECEIVED:", trace, "RESOLVED:", trace_arg)

    if background is not None:
        background.add_task(_train_background, trace_arg, epochs, out_name)
    else:
        threading.Thread(target=_train_background, args=(trace_arg, epochs, out_name), daemon=True).start()

    return {"status": "started", "trace": trace_arg, "epochs": epochs, "out_name": out_name}

@app.get("/api/models")
def api_list_models():
    return list_models()

# --- Evaluate: now supports TTL/LRU/LFU/DRL + extra knobs ---
@app.post("/api/models/evaluate")
def api_evaluate_model(payload: dict):
    """
    Evaluate a model on a given trace (synchronous).
    Request: { model_id?: str, model_path?: str, trace?: str, out?: str, plot?: bool,
               max_bytes?: int, ttl_s?: int, hit_ms?: int }
    Response: { out: "<csv>", rows: [...], plot?: "<png>", max_bytes: int }
    """
    model_id = payload.get("model_id")
    trace_raw = payload.get("trace", "simulator/data/run_1.csv")
    out_csv = payload.get("out") or str(RL_LOGS_DIR / f"eval_{int(time.time())}.csv")
    want_plot = bool(payload.get("plot", False))

    # ---- new: tunables (fallback to env/defaults) ----
    try:
        max_bytes = int(payload.get("max_bytes")
                        or os.getenv("CACHE_MAX_BYTES")
                        or MAX_BYTES)
    except Exception:
        max_bytes = MAX_BYTES

    ttl_s = int(payload.get("ttl_s", int(os.getenv("CACHE_DEFAULT_TTL_S", DEFAULT_TTL))))
    hit_ms = int(payload.get("hit_ms", int(os.getenv("CACHE_HIT_LATENCY_MS", HIT_MS))))

    # resolve model path
    model_path = None
    model_entry = None
    if model_id:
        model_entry = find_model_by_id(str(model_id))
        if not model_entry:
            raise HTTPException(status_code=404, detail="Model id not found")
        model_path = model_entry.get("path")
    else:
        model_path = payload.get("model_path")

    if not model_path:
        raise HTTPException(status_code=400, detail="model_id or model_path required")
    if not Path(model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model file not found: {model_path}")

    # resolve trace path like training
    trace_path = None
    for candidate in [Path(trace_raw), REPO_ROOT / trace_raw, Path(__file__).resolve().parents[1] / trace_raw]:
        if candidate.exists():
            trace_path = candidate
            break
    if trace_path is None:
        raise HTTPException(status_code=400, detail=f"Trace file not found (tried): {trace_raw}")

    # ensure logs dir
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # run evaluator as module; pass max-bytes/ttl/hit-ms and optional plot
    cmd = [
        PY_EXE, "-m", "rl_agent.evaluate",
        "--trace", str(trace_path),
        "--model", str(model_path),
        "--out", str(out_path),
        "--max-bytes", str(max_bytes),
        "--ttl-s", str(ttl_s),
        "--hit-ms", str(hit_ms),
    ]
    plot_path = None
    if want_plot:
        plot_path = str(out_path.with_suffix(".png"))
        cmd += ["--plot", "--plot-save", plot_path]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    print("Starting eval subprocess:", " ".join(cmd))
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {e}")

    # parse CSV -> rows
    try:
        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, Any]] = []
            for r in reader:
                def num(x):
                    try:
                        if x is None or str(x).strip() == "":
                            return None
                        if str(x).isdigit():
                            return int(x)
                        return float(x)
                    except Exception:
                        return x
                rows.append({
                    "policy": r.get("policy"),
                    "total": num(r.get("total")),
                    "hits": num(r.get("hits")),
                    "hit_ratio_pct": num(r.get("hit_ratio_pct")),
                    "avg_latency_ms": num(r.get("avg_latency_ms")),
                    "stale_pct": num(r.get("stale_pct")),
                    "bytes_from_cache": num(r.get("bytes_from_cache")),
                    "bytes_from_origin": num(r.get("bytes_from_origin")),
                    "bandwidth_savings_pct": num(r.get("bandwidth_savings_pct")),
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read eval CSV: {e}")

    # update registry.last_eval
    if model_entry:
        entries = load_registry()
        for e in entries:
            if str(e.get("id")) == str(model_entry.get("id")):
                e["last_eval"] = {
                    "csv": str(out_path),
                    "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
                }
                break
        save_registry(entries)

    resp: Dict[str, Any] = {"out": str(out_path), "rows": rows, "max_bytes": int(max_bytes)}
    if plot_path:
        resp["plot"] = plot_path
    return resp

# --- Report helpers (JSON/CSV/Plot) ---
def _resolve_last_eval_csv(model_id: Optional[str]) -> Path:
    entries = load_registry()
    chosen = None
    last_eval = None

    if model_id:
        chosen = find_model_by_id(model_id)
        if not chosen:
            raise HTTPException(status_code=404, detail="Model id not found")
        last_eval = chosen.get("last_eval")
        if not last_eval:
            raise HTTPException(status_code=404, detail="No evaluation recorded for this model")
    else:
        candidates = [e for e in entries if e.get("last_eval")]
        if not candidates:
            raise HTTPException(status_code=404, detail="No evaluations recorded in registry")
        candidates.sort(key=lambda x: x["last_eval"].get("ts", ""), reverse=True)
        chosen = candidates[0]
        last_eval = chosen.get("last_eval")

    csv_path = Path(last_eval["csv"])
    if not csv_path.exists():
        alt = RL_LOGS_DIR / csv_path.name
        if alt.exists():
            csv_path = alt
        else:
            raise HTTPException(status_code=404, detail=f"Evaluation CSV not found at {csv_path}")
    return csv_path

@app.get("/api/models/report")
def api_get_last_eval_report(model_id: Optional[str] = None):
    csv_path = _resolve_last_eval_csv(model_id)

    # best-effort model metadata (for ts)
    entries = load_registry()
    chosen = find_model_by_id(model_id) if model_id else None
    if chosen is None:
        candidates = [e for e in entries if e.get("last_eval")]
        candidates.sort(key=lambda x: x["last_eval"].get("ts", ""), reverse=True)
        chosen = candidates[0] if candidates else None

    try:
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            def _convert_row(r: Dict[str, str]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for k, v in r.items():
                    if v is None:
                        out[k] = v
                        continue
                    s = v.strip()
                    try:
                        if s == "":
                            out[k] = None
                        elif s.isdigit():
                            out[k] = int(s)
                        else:
                            out[k] = float(s)
                    except Exception:
                        out[k] = v
                return out
            rows_converted = [_convert_row(r) for r in reader]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read eval CSV: {e}")

    ts = None
    if chosen and chosen.get("last_eval"):
        ts = chosen["last_eval"].get("ts")

    return {"model": chosen, "csv": str(csv_path), "ts": ts, "rows": rows_converted}

@app.get("/api/models/report/csv")
def api_report_csv(model_id: Optional[str] = None):
    csv_path = _resolve_last_eval_csv(model_id)
    return FileResponse(path=str(csv_path), media_type="text/csv", filename=csv_path.name)

@app.get("/api/models/report/plot")
def api_report_plot(model_id: Optional[str] = None):
    """
    Returns/creates a PNG plot for the last evaluation CSV (bar chart).
    Now supports multiple policies (TTL/LRU/LFU/DRL).
    """
    csv_path = _resolve_last_eval_csv(model_id)
    png_path = csv_path.with_suffix(".png")

    if not png_path.exists():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

            # Build bars for hit ratio for whatever policies are present
            policies = [r.get("policy", "").upper() for r in rows]
            hits = []
            for r in rows:
                val = r.get("hit_ratio_pct", "0")
                try:
                    hits.append(float(val))
                except Exception:
                    hits.append(0.0)

            fig = plt.figure()
            plt.title("Hit Ratio (%)")
            plt.bar(policies, hits)
            fig.savefig(png_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not generate plot: {e}")

    return FileResponse(path=str(png_path), media_type="image/png", filename=png_path.name)

# --- Cache/Stats/History/Experiments/Simulate ---
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

@app.get("/api/cache/stats")
def cache_stats(db: Session = Depends(get_db)):
    count_val = db.query(func.count(models.CacheItem.id)).scalar()
    try:
        items = int(count_val or 0)
    except Exception:
        items = int(float(count_val or 0))

    used = total_cache_bytes(db)

    # Normalize env to int
    try:
        max_bytes = int(os.getenv("CACHE_MAX_BYTES", MAX_BYTES))
    except Exception:
        max_bytes = int(MAX_BYTES or 0)

    if max_bytes <= 0:
        pct_full = 0.0
    else:
        pct_full = round(100.0 * (float(used) / float(max_bytes)), 2)

    return {
        "items": items,
        "bytes_used": int(used),
        "max_bytes": int(max_bytes),
        "pct_full": pct_full,
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
        {
            "id": r.id,
            "started_ts": r.started_ts.isoformat(),
            "workload": r.workload,
            "minutes": r.minutes,
            "rps": r.rps,
            "rate": r.rate,
            "status": r.status,
        } for r in rows
    ]

# --- Per-run sampler: collects metrics only for this run_id/time window ---
from threading import Event

def sample_run_metrics(run_id: int, started_ts: datetime, stop_event: Event):
    """Every second, compute metrics for requests in [started_ts, now] and
    store StatSnapshot rows tagged with run_id. Stops when stop_event is set."""
    hz = SAMPLE_HZ or 1
    period = 1.0 / float(hz)

    with SessionLocal() as db:
        while not stop_event.is_set():
            now_cutoff = datetime.now(timezone.utc)

            # Metrics restricted to this run’s time window by joining Request.ts
            hit_avg = (
                db.query(func.avg(case((models.Outcome.cache_hit == True, 1), else_=0)))
                  .join(models.Request, models.Outcome.request_id == models.Request.id)
                  .filter(models.Request.ts >= started_ts, models.Request.ts <= now_cutoff)
                  .scalar()
                or 0.0
            )
            avg_latency = (
                db.query(func.avg(models.Outcome.served_latency_ms))
                  .join(models.Request, models.Outcome.request_id == models.Request.id)
                  .filter(models.Request.ts >= started_ts, models.Request.ts <= now_cutoff)
                  .scalar()
                or 0.0
            )
            stale_avg = (
                db.query(func.avg(case((models.Outcome.staleness_s > 0, 1), else_=0)))
                  .join(models.Request, models.Outcome.request_id == models.Request.id)
                  .filter(models.Request.ts >= started_ts, models.Request.ts <= now_cutoff)
                  .scalar()
                or 0.0
            )

            snap = models.StatSnapshot(
                run_id=run_id,
                hit_ratio_pct=round(100.0 * float(hit_avg), 2),
                avg_latency_ms=round(float(avg_latency), 2),
                staleness_pct=round(100.0 * float(stale_avg), 2),
            )
            db.add(snap)
            db.commit()
            time.sleep(period)


@app.post("/api/experiments/run")
def run_experiment(payload: dict, db: Session = Depends(get_db)):
    # Inputs with safe defaults (preserve existing behavior if UI untouched)
    workload = payload.get("workload", "zipf")
    minutes = int(payload.get("minutes", 2))
    rps = int(payload.get("rps", 5))
    rate = int(payload.get("rate", 10))

    # NEW: bigger keyspace & reproducibility
    objects = int(payload.get("objects", 200))     # try 5000–50000 for PhD runs
    clients = int(payload.get("clients", 50))
    seed = payload.get("seed")                     # None or an int

    run = models.Run(
        workload=workload, minutes=minutes, rps=rps, rate=rate, status="running"
    )
    db.add(run); db.commit(); db.refresh(run)

    csv_path = os.path.abspath(os.path.join(SIM_DIR, f"data/run_{run.id}.csv"))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"[DEBUG] Preparing experiment for run_id={run.id}")
    print(f"[DEBUG] SIM_DIR = {SIM_DIR}")
    print(f"[DEBUG] CSV_PATH = {csv_path}")

    make_cmd = [
        PY_EXE, os.path.join(SIM_DIR, "make_csv.py"),
        "--workload", workload,
        "--minutes", str(minutes),
        "--rps", str(rps),
        "--objects", str(objects),
        "--clients", str(clients),
        "--outfile", csv_path,
    ]
    # Pass seed if your make_csv.py supports it (optional & no-op otherwise)
    seed = payload.get("seed")
    if seed is not None and str("seed").strip() != "":
        make_cmd.extend(["--seed", str(seed)])

    replay_cmd = [
        PY_EXE, os.path.join(SIM_DIR, "replay.py"),
        "--file", csv_path,
        "--base", "http://127.0.0.1:8000",
        "--rate", str(rate),
    ]

    print(f"[DEBUG] make_cmd = {' '.join(make_cmd)}")
    print(f"[DEBUG] replay_cmd = {' '.join(replay_cmd)}")

    # NEW: per-run sampler thread
    stop_event = Event()
    sampler = threading.Thread(
        target=sample_run_metrics, args=(run.id, run.started_ts, stop_event), daemon=True
    )
    sampler.start()

    def worker():
        try:
            print("[RUNNER] Running make_csv...")
            # Use subprocess.run with a timeout guard (minutes + 30s cushion)
            subprocess.run(make_cmd, check=True, timeout=(minutes * 60 + 30))

            print("[RUNNER] Running replay...")
            subprocess.run(replay_cmd, check=True, timeout=(minutes * 60 + 120))

            run.status = "done"
            print(f"[RUNNER] ✅ Run {run.id} completed successfully.")
        except subprocess.TimeoutExpired as te:
            run.status = "timeout"
            print(f"[ERROR] Run {run.id} timed out: {te}")
        except Exception as e:
            run.status = "error"
            print(f"[ERROR] Subprocess failed: {e}")
        finally:
            # stop per-run sampler
            stop_event.set()
            with SessionLocal() as s:
                s.merge(run); s.commit()
                print(f"[RUNNER] ✅ Status stored in database: {run.status}")

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

        q = models.Request(
            ts=now_utc(),
            client_id="sim",
            object_id=object_id,
            object_size_bytes=size_bytes,
            origin_latency_ms=served_latency if not hit else HIT_MS,
            was_write=False,
        )
        db.add(q)
        db.flush()

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