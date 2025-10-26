# backend/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# Load .env next to backend/main.py
load_dotenv(BASE_DIR / ".env")

def require_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None or (isinstance(val, str) and val.strip() == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return val

def resolve_path(maybe: str) -> Path:
    """
    Resolve a file/directory path tried in this order:
      1) as-is
      2) REPO_ROOT / maybe
      3) BASE_DIR / maybe
    Returns absolute Path.
    """
    candidates = [
        Path(maybe),
        REPO_ROOT / maybe,
        BASE_DIR / maybe,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # return best-guess absolute even if missing (caller can check exists)
    return (REPO_ROOT / maybe).resolve()
