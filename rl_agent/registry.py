# rl_agent/registry.py
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

def ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text("[]")  # empty list

def list_models() -> List[Dict[str, Any]]:
    ensure_models_dir()
    with open(REGISTRY_PATH, "r") as f:
        try:
            data = json.load(f)
            return data
        except Exception:
            return []

def add_model_entry(file_path: str, meta: Dict[str, Any] = None) -> Dict[str, Any]:
    ensure_models_dir()
    meta = meta or {}
    entry = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "path": str(Path(file_path).resolve()),
        "created_ts": datetime.utcnow().isoformat() + "Z",
        "meta": meta
    }
    data = list_models()
    data.append(entry)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return entry

def find_model_by_id(mid: str):
    for m in list_models():
        if m["id"] == mid:
            return m
    return None
