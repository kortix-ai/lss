# lss_config.py
import copy as _copy
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

# ── Embedding provider ───────────────────────────────────────────────────────
#
# "openai"  — OpenAI text-embedding-3-small (256d), requires OPENAI_API_KEY
# "local"   — fastembed bge-small-en-v1.5 (384d), 100% offline, ~125MB download
#
# Auto-detection: if OPENAI_API_KEY is set, default to "openai".
# If not set but fastembed is installed, default to "local".
# Explicit override via LSS_PROVIDER env var or `lss config provider <name>`.

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
OPENAI_DIM = int(os.getenv("OPENAI_DIM", "256"))
LOCAL_MODEL = os.getenv("LSS_LOCAL_MODEL", "BAAI/bge-small-en-v1.5")
LOCAL_DIM = int(os.getenv("LSS_LOCAL_DIM", "384"))

def _detect_provider() -> str:
    """Auto-detect embedding provider based on available resources."""
    # 1. Explicit env var overrides everything
    env_provider = os.getenv("LSS_PROVIDER", "").strip().lower()
    if env_provider in ("openai", "local"):
        return env_provider

    # 2. Check persistent config
    if CONFIG_FILE.exists():
        try:
            data = _json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            cfg_provider = data.get("embedding_provider", "").strip().lower()
            if cfg_provider in ("openai", "local"):
                return cfg_provider
        except (_json.JSONDecodeError, OSError):
            pass

    # 3. Auto-detect: prefer openai if API key is set, else try local
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"

    try:
        import fastembed  # noqa: F401
        return "local"
    except ImportError:
        pass

    # 4. Fallback to openai (will fail later with a helpful message)
    return "openai"


EMBEDDING_PROVIDER = _detect_provider()

def _provider_model_dim():
    """Return (model_name, dim) for the active provider."""
    if EMBEDDING_PROVIDER == "local":
        return LOCAL_MODEL, LOCAL_DIM
    return OPENAI_MODEL, OPENAI_DIM


# Version key source of truth — incorporates provider so switching
# openai <-> local triggers re-embedding (BM25 index stays intact).
PREPROC_VER = 2
CHUNKER_VER = 4
_model, _dim = _provider_model_dim()
VERSION_KEY = f"{_model}:{_dim}:p{PREPROC_VER}:c{CHUNKER_VER}"

# Global debug flag (set by CLI)
DEBUG = False

def set_debug(enabled: bool):
    global DEBUG
    DEBUG = enabled


# ── Persistent config (watch paths, exclusions) ─────────────────────────────

_DEFAULT_CONFIG = {
    "watch_paths": [],
    "exclude_patterns": [],
    "include_extensions": [],
    "embedding_provider": "",  # "" = auto-detect
}


def load_config() -> dict:
    """Load ~/.lss/config.json, returning defaults if missing."""
    if CONFIG_FILE.exists():
        try:
            data = _json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            # Merge with defaults so new keys are always present
            merged = _copy.deepcopy(_DEFAULT_CONFIG)
            merged.update(data)
            return merged
        except (_json.JSONDecodeError, OSError):
            pass
    return _copy.deepcopy(_DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    """Persist config to ~/.lss/config.json."""
    LSS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        _json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
