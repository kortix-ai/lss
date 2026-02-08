# lss_config.py
import json as _json
import os, sys
from pathlib import Path

def _is_frozen() -> bool:
    return bool(
        getattr(sys, "frozen", False) or
        os.getenv("NUITKA_ONEFILE_PARENT") or
        getattr(sys, "_MEIPASS", None) or
        os.getenv("_MEIPASS2")
    )

# Mode: dev | test | prod | build
LSS_ENV = (os.getenv("LSS_ENV") or ("build" if _is_frozen() else "dev")).lower()

def _load_dotenv(paths):
    for base in paths:
        p = Path(base).expanduser() / ".env"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k not in os.environ:            # no overrides
                    os.environ[k] = v
            break

# Only load .env in dev/test
if LSS_ENV in ("dev", "test"):
    lss_dir = os.getenv("LSS_DIR", "~/.lss")
    _load_dotenv([
        os.getcwd(),
        lss_dir,
        "~/.config/lss",
        os.path.dirname(__file__),
    ])

# Shared config
LSS_DIR = Path(os.getenv("LSS_DIR", "~/.lss")).expanduser()
LSS_DB = LSS_DIR / "lss.db"
CONFIG_FILE = LSS_DIR / "config.json"

# Version key source of truth
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
OPENAI_DIM = int(os.getenv("OPENAI_DIM", "256"))
PREPROC_VER = 2
CHUNKER_VER = 4
VERSION_KEY = f"{OPENAI_MODEL}:{OPENAI_DIM}:p{PREPROC_VER}:c{CHUNKER_VER}"

# Global debug flag (set by CLI)
DEBUG = False

def set_debug(enabled: bool):
    global DEBUG
    DEBUG = enabled


# ── Persistent config (watch paths, exclusions) ─────────────────────────────

_DEFAULT_CONFIG = {
    "watch_paths": [],
    "exclude_patterns": [],
}


def load_config() -> dict:
    """Load ~/.lss/config.json, returning defaults if missing."""
    if CONFIG_FILE.exists():
        try:
            data = _json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            # Merge with defaults so new keys are always present
            merged = dict(_DEFAULT_CONFIG)
            merged.update(data)
            return merged
        except (_json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    """Persist config to ~/.lss/config.json."""
    LSS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        _json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
