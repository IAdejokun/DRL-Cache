#!/usr/bin/env python3
"""
e2e_runner.py

One-file end-to-end runner:
  - trains a model with rl_agent/train_offline.py
  - registers it via rl_agent/registry.add_model_entry
  - evaluates it with rl_agent/evaluate.py (CSV + saved plot)
  - starts backend (uvicorn) in background (cwd=backend)
  - triggers POST /api/simulate?mode=drl and saves JSON
  - optionally prints DB counts via psql if DATABASE_URL present

Usage (from repo root):
  python scripts/e2e_runner.py --trace simulator/data/run_1.csv --epochs 10

Notes:
  - This script calls your existing train/evaluate Python scripts as subprocesses
    to isolate logs and avoid interfering with uvicorn.
  - It tries to be cross-platform. Use an activated venv (recommended).
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
import argparse

REPO_ROOT = Path.cwd()
RL_AGENT_DIR = REPO_ROOT / "rl_agent"
BACKEND_DIR = REPO_ROOT / "backend"
LOGS_DIR = RL_AGENT_DIR / "logs"
MODELS_DIR = RL_AGENT_DIR / "models"
REGISTRY_PY = RL_AGENT_DIR / "registry.py"  # optional import

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable  # ensures the same Python interpreter is used

def timestamp():
    return int(time.time())

def run_subprocess(cmd, logfile_path=None, env=None, cwd=None):
    """Run a subprocess and stream output to logfile_path (if provided)."""
    env = env or os.environ.copy()
    print("Running:", " ".join(cmd))
    if logfile_path:
        with open(logfile_path, "ab") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=cwd, check=False)
            return proc.returncode
    else:
        proc = subprocess.run(cmd, env=env, cwd=cwd, check=False)
        return proc.returncode

def train_model(trace, epochs, out_path):
    """Call train_offline.py to train a model and write to out_path."""
    log_file = LOGS_DIR / f"train_{timestamp()}.log"
    cmd = [PY, str(RL_AGENT_DIR / "train_offline.py"),
           "--trace", str(trace),
           "--epochs", str(epochs),
           "--out", str(out_path)]
    print("=== TRAINING ===")
    rc = run_subprocess(cmd, logfile_path=log_file, cwd=REPO_ROOT)
    if rc != 0:
        print(f"[WARN] Training subprocess exited with code {rc}. Check log: {log_file}")
    else:
        print(f"[OK] Training finished. Log: {log_file}")
    return log_file, rc

def register_model(path, trace, epochs):
    """Register model via rl_agent.registry.add_model_entry if available."""
    # Insert repo root to sys.path for imports
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from rl_agent import registry
        entry = registry.add_model_entry(str(path), meta={"trace": str(trace), "epochs": epochs})
        print("Registered model:", entry)
        return entry
    except Exception as e:
        print("Could not register model via rl_agent.registry:", e)
        print("You can register manually by calling registry.add_model_entry() later.")
        return None

def evaluate_model(trace, model_path, out_csv, plot_path):
    """Call evaluate.py to evaluate model and produce CSV + saved plot (uses MPL backend=Agg for safety)."""
    log_file = LOGS_DIR / f"eval_{timestamp()}.log"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # avoid opening GUI windows
    cmd = [PY, str(RL_AGENT_DIR / "evaluate.py"),
           "--trace", str(trace),
           "--model", str(model_path),
           "--out", str(out_csv),
           "--plot",  # instruct evaluate.py to run plotting
           "--plot-save", str(plot_path)]
    print("=== EVALUATING ===")
    rc = run_subprocess(cmd, logfile_path=log_file, env=env, cwd=REPO_ROOT)
    if rc != 0:
        print(f"[WARN] Eval subprocess exited with code {rc}. Check log: {log_file}")
    else:
        print(f"[OK] Evaluation finished. CSV: {out_csv}  Plot: {plot_path}")
    return log_file, rc

def start_backend(log_path=None, port=8000):
    """Start uvicorn main:app in background; returns Popen object."""
    print("=== STARTING BACKEND ===")
    env = os.environ.copy()
    cmd = [PY, "-m", "uvicorn", "main:app", "--port", str(port)]
    # Run server as background process
    stdout = open(log_path, "ab") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, cwd=str(BACKEND_DIR), env=env, stdout=stdout, stderr=subprocess.STDOUT)
    print(f"Started uvicorn (pid={proc.pid}), logs -> {log_path}")
    return proc

def wait_for_health(url="http://127.0.0.1:8000/health", timeout=60):
    """Wait up to `timeout` seconds for backend /health to respond."""
    print("Waiting for backend health endpoint...", end="", flush=True)
    start = time.time()
    while True:
        try:
            with urlopen(url, timeout=2) as r:
                body = r.read().decode("utf-8")
                print("\nBackend healthy:", body)
                return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
        if time.time() - start > timeout:
            print("\nTimed out waiting for backend health.")
            return False

def trigger_simulate(mode="drl", out_json=None):
    """POST to /api/simulate?mode=... and save JSON to out_json."""
    url = f"http://127.0.0.1:8000/api/simulate?{urlencode({'mode': mode})}"
    print("Triggering simulate:", url)
    req = Request(url, method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
            j = json.loads(data)
            if out_json:
                Path(out_json).write_text(json.dumps(j, indent=2))
            print("Simulate response:", j)
            return j
    except HTTPError as e:
        print("HTTP Error:", e.code, e.reason)
    except URLError as e:
        print("URL Error:", e.reason)
    except Exception as e:
        print("Simulate request failed:", e)
    return None

def print_db_counts():
    """Attempt to print DB counts using psql if DATABASE_URL present in backend/.env or env."""
    # look for DATABASE_URL
    dburl = None
    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                dburl = line.split("=", 1)[1].strip()
                break
    if not dburl:
        dburl = os.environ.get("DATABASE_URL")

    if not dburl:
        print("DATABASE_URL not found in backend/.env or environment. Skipping DB counts.")
        return

    psql_path = shutil.which("psql")
    if not psql_path:
        print("psql not found on PATH. To inspect DB run these queries manually (in pgAdmin/psql):")
        print("  SELECT count(*) FROM requests;")
        print("  SELECT count(*) FROM outcomes;")
        return

    print("Querying DB counts using psql (this will prompt for auth if required)...")
    try:
        # psql accepts -d <connstring> then -c commands
        subprocess.run([psql_path, dburl, "-c", "SELECT 'requests' as table, count(*) FROM requests;"], check=False)
        subprocess.run([psql_path, dburl, "-c", "SELECT 'outcomes' as table, count(*) FROM outcomes;"], check=False)
    except Exception as e:
        print("psql queries failed:", e)

def main():
    p = argparse.ArgumentParser(description="End-to-end runner: train -> eval -> backend -> simulate")
    p.add_argument("--trace", default="simulator/data/run_1.csv", help="Path to trace CSV")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--start-backend", action="store_true", help="Start backend after training (default: false)")
    p.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    p.add_argument("--port", type=int, default=8000, help="Backend port (default 8000)")
    args = p.parse_args()

    trace = Path(args.trace)
    if not trace.exists():
        print(f"[ERROR] Trace file not found: {trace}")
        sys.exit(2)

    ts = timestamp()
    model_out = MODELS_DIR / f"model_{ts}.pt"
    eval_csv = LOGS_DIR / f"eval_{ts}.csv"
    plot_png = LOGS_DIR / f"compare_{ts}.png"
    sim_json = LOGS_DIR / f"sim_{ts}.json"
    backend_log = REPO_ROOT / f"backend_uvicorn_{ts}.log"

    # 1) Train
    train_log, rc = train_model(trace, args.epochs, model_out)
    if rc != 0:
        print("[WARN] Training returned non-zero exit code. You can inspect the train log. Proceeding to registration/eval at your risk.")

    # 2) Register model
    if model_out.exists():
        register_model(model_out, trace, args.epochs)
    else:
        print("[WARN] Model file not found after training:", model_out)

    # 3) Evaluate
    if not args.no_eval:
        eval_log, rc_eval = evaluate_model(trace, model_out, eval_csv, plot_png)
        if rc_eval != 0:
            print("[WARN] Evaluation had non-zero exit code. Check eval log:", eval_log)
    else:
        print("Skipping evaluation (--no-eval set).")

    # 4) Start backend (optional)
    backend_proc = None
    if args.start_backend:
        backend_proc = start_backend(log_path=str(backend_log), port=args.port)
        # wait for /health
        ok = wait_for_health(url=f"http://127.0.0.1:{args.port}/health", timeout=60)
        if not ok:
            print("Backend did not become healthy. Check log:", backend_log)
    else:
        print("Skipping backend start (use --start-backend to start uvicorn here).")

    # 5) Trigger simulate (only if backend is up)
    if args.start_backend:
        sim_resp = trigger_simulate(mode="drl", out_json=str(sim_json))
    else:
        print("Not triggering simulate because backend wasn't started by this script.")

    # 6) DB counts
    print_db_counts()

    print("\n=== SUMMARY ===")
    print("Model out:", model_out)
    print("Train log:", train_log)
    print("Eval CSV:", eval_csv)
    print("Plot PNG:", plot_png)
    print("Sim JSON:", sim_json)
    if backend_proc:
        print("Backend pid:", backend_proc.pid)
        print("Backend log:", backend_log)
        print("To stop backend: kill PID or Ctrl+C the process (if you started it manually).")

    print("Done.")

if __name__ == "__main__":
    main()
