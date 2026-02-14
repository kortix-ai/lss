"""Shared fixtures for the lss test suite."""

import os
import sys
import pytest
from pathlib import Path

# Ensure the lss package root is on sys.path so we can import lss_* modules
_LSS_ROOT = Path(__file__).resolve().parent.parent
if str(_LSS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LSS_ROOT))


@pytest.fixture(autouse=True)
def isolated_lss_dir(tmp_path, monkeypatch):
    """Every test gets its own LSS_DIR so tests don't share state."""
    lss_dir = tmp_path / ".lss"
    lss_dir.mkdir()
    monkeypatch.setenv("LSS_DIR", str(lss_dir))
    monkeypatch.setenv("LSS_ENV", "test")
    monkeypatch.setenv("OAI_TIMEOUT", "15")

    # Update the cached OAI_TIMEOUT in semantic_search if already imported
    try:
        import semantic_search
        semantic_search.OAI_TIMEOUT = 15.0
    except (ImportError, AttributeError):
        pass

    # Force config module to pick up new LSS_DIR
    import lss_config

    lss_config.LSS_DIR = lss_dir
    lss_config.LSS_DB = lss_dir / "lss.db"
    lss_config.CONFIG_FILE = lss_dir / "config.json"

    # Preserve embedding provider globals so tests that call
    # `config provider` or `set_embedding_provider()` don't leak state.
    monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", lss_config.EMBEDDING_PROVIDER)
    monkeypatch.setattr(lss_config, "VERSION_KEY", lss_config.VERSION_KEY)

    # Clear any cached DB connections / file caches
    import lss_store

    lss_store._file_cache.clear()
    # Also update the module-level LSS_DIR/LSS_DB that lss_store imported at
    # load time so _init_db() creates the DB in the right place.
    lss_store.LSS_DIR = lss_dir
    lss_store.LSS_DB = lss_dir / "lss.db"

    yield lss_dir


@pytest.fixture
def sample_dir(tmp_path):
    """Create a temp directory with sample files for indexing."""
    d = tmp_path / "sample"
    d.mkdir()

    (d / "readme.md").write_text(
        "Kubernetes uses containers for deployment with GitHub Actions "
        "CI/CD pipelines and ArgoCD for GitOps workflows"
    )
    (d / "notes.txt").write_text(
        "The authentication system uses JWT tokens with RSA-256 signing "
        "for secure API access"
    )
    (d / "code.py").write_text(
        'def hello():\n    """Greet the user."""\n    return "Hello, world!"\n'
    )
    (d / "data.json").write_text(
        '{"project": "lss", "description": "Local semantic search engine"}'
    )
    # Binary file — should be detected as non-text
    (d / "binary.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
    )
    # Nested directory
    nested = d / "deep" / "nested"
    nested.mkdir(parents=True)
    (nested / "file.md").write_text(
        "This is a deeply nested markdown file about database migrations"
    )
    # Hidden file — should be skipped during directory ingest
    (d / ".hidden_file").write_text("This is a hidden file that should be ignored")
    # node_modules — should be excluded
    nm = d / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}")

    return d


@pytest.fixture
def require_openai():
    """Skip the test when OPENAI_API_KEY is not set."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
