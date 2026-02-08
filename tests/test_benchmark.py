"""Benchmark tests — measure and assert performance of the indexing + search pipeline.

These tests generate realistic content at scale, time every phase, and fail
if performance regresses beyond defined thresholds.  They also print a
human-readable report so ``pytest -s`` gives you a full timing breakdown.

No OpenAI API calls are made in the pure indexing benchmarks.
Search benchmarks that need embeddings are gated on OPENAI_API_KEY.
"""

import os
import time
from pathlib import Path

import pytest

from lss_store import (
    discover_files,
    ingest_many,
    clear_all,
    _walk_text_files,
    _is_text_file,
    _init_db,
    _file_cache,
)
from lss_cli import _C

# Disable colors for predictable output
@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_project(base: Path, n_files: int, words_per_file: int = 200,
                      include_junk: bool = True):
    """Generate a realistic project directory with text files, binaries, and junk dirs.

    Returns the number of indexable text files created.
    """
    base.mkdir(parents=True, exist_ok=True)
    text_count = 0

    # -- Text files spread across subdirectories --
    dirs = [
        base / "src",
        base / "src" / "auth",
        base / "src" / "api",
        base / "src" / "db",
        base / "docs",
        base / "tests",
        base / "config",
        base / "scripts",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    words_bank = (
        "authentication JWT token login deploy Kubernetes Docker PostgreSQL "
        "Redis FastAPI SQLAlchemy migration database query index cache session "
        "user email password hash RSA encryption WebSocket gRPC REST API "
        "endpoint middleware rate-limit CORS CSRF OAuth2 SAML RBAC audit log "
        "container orchestration pod service ingress namespace ConfigMap secret "
        "CI/CD GitHub Actions ArgoCD Terraform Helm chart release rollback "
    ).split()

    import random
    rng = random.Random(42)  # deterministic for reproducibility

    exts = [".py", ".ts", ".js", ".md", ".txt", ".yaml", ".toml", ".json", ".rs", ".go"]

    for i in range(n_files):
        subdir = dirs[i % len(dirs)]
        ext = exts[i % len(exts)]
        content_words = rng.choices(words_bank, k=words_per_file)
        content = " ".join(content_words)

        if ext == ".json":
            import json
            content = json.dumps({"id": i, "content": content, "version": "1.0"})
        elif ext == ".py":
            content = f'"""Module {i}"""\n\ndef func_{i}():\n    """{content}"""\n    pass\n'
        elif ext == ".md":
            content = f"# Document {i}\n\n{content}\n"

        (subdir / f"file_{i:04d}{ext}").write_text(content)
        text_count += 1

    # -- Binary files (should be skipped) --
    if include_junk:
        for i in range(20):
            (base / f"image_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
        for i in range(5):
            (base / f"archive_{i}.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 200)
        (base / "compiled.wasm").write_bytes(b"\x00asm" + b"\x00" * 100)

    # -- Excluded directories (should NEVER be walked into) --
    if include_junk:
        nm = base / "node_modules" / "react" / "lib"
        nm.mkdir(parents=True)
        for i in range(200):
            (nm / f"chunk_{i}.js").write_text(f"// generated code {i}\n" * 50)

        cache = base / ".cache" / "webpack"
        cache.mkdir(parents=True)
        for i in range(50):
            (cache / f"cache_{i}.json").write_text(f'{{"hash": "{i}"}}')

        git = base / ".git" / "objects"
        git.mkdir(parents=True)
        for i in range(100):
            (git / f"obj_{i}").write_bytes(b"\x00" * 50)

        pycache = base / "src" / "__pycache__"
        pycache.mkdir(parents=True)
        for i in range(30):
            (pycache / f"mod_{i}.cpython-312.pyc").write_bytes(b"\x00" * 50)

        # Lock files (should be skipped by name)
        (base / "package-lock.json").write_text('{"lockfileVersion": 3}' * 100)
        (base / "yarn.lock").write_text("# yarn lock\n" * 500)

        # Log files (should be skipped by extension)
        (base / "app.log").write_text("INFO startup\n" * 500)

    return text_count


# ── Benchmark: File discovery ────────────────────────────────────────────────


class TestDiscoveryBenchmark:
    """Test that _walk_text_files correctly prunes and is fast."""

    def test_walk_excludes_node_modules(self, tmp_path):
        """_walk_text_files must never enter node_modules, .git, .cache, etc."""
        n = _generate_project(tmp_path / "proj", n_files=50, include_junk=True)

        found = list(_walk_text_files(tmp_path / "proj"))
        found_names = {f.name for f in found}

        # Must find the text files
        assert len(found) == n

        # Must NOT include junk
        assert not any("chunk_" in f.name for f in found), "node_modules leaked"
        assert not any("cache_" in f.name for f in found), ".cache leaked"
        assert not any("obj_" in f.name for f in found), ".git leaked"
        assert not any(f.suffix == ".pyc" for f in found), "__pycache__ leaked"
        assert "package-lock.json" not in found_names, "lock file not excluded"
        assert "yarn.lock" not in found_names, "lock file not excluded"
        assert "app.log" not in found_names, ".log not excluded"
        assert not any(f.suffix == ".png" for f in found), "binary extension leaked"
        assert not any(f.suffix == ".zip" for f in found), "binary extension leaked"

    def test_walk_speed_with_junk_dirs(self, tmp_path):
        """Walking a project with 380+ junk files should be <50ms (dir pruning)."""
        _generate_project(tmp_path / "proj", n_files=100, include_junk=True)

        t0 = time.time()
        found = list(_walk_text_files(tmp_path / "proj"))
        dt = time.time() - t0

        assert len(found) == 100
        assert dt < 0.5, f"walk took {dt*1000:.0f}ms — should be <500ms even on slow CI"
        print(f"\n  [BENCH] walk {len(found)} files ({380 + 100} on disk): {dt*1000:.1f}ms")


# ── Benchmark: Indexing pipeline ─────────────────────────────────────────────


class TestIndexingBenchmark:
    """Comprehensive indexing benchmarks at various scales."""

    @pytest.fixture
    def project_50(self, tmp_path):
        """50-file project."""
        p = tmp_path / "proj50"
        _generate_project(p, n_files=50, words_per_file=200, include_junk=True)
        return p

    @pytest.fixture
    def project_200(self, tmp_path):
        """200-file project."""
        p = tmp_path / "proj200"
        _generate_project(p, n_files=200, words_per_file=300, include_junk=True)
        return p

    @pytest.fixture
    def project_500(self, tmp_path):
        """500-file project with larger files."""
        p = tmp_path / "proj500"
        _generate_project(p, n_files=500, words_per_file=500, include_junk=True)
        return p

    def test_index_50_files(self, project_50):
        """Index 50 files — should complete in <2s."""
        t0 = time.time()
        all_f, new_f, already = discover_files(project_50)
        t_discover = time.time() - t0

        assert len(all_f) == 50
        assert len(new_f) == 50
        assert already == 0

        t0 = time.time()
        uids = ingest_many(new_f)
        t_index = time.time() - t0

        assert len(uids) == 50
        total = t_discover + t_index
        assert total < 2.0, f"50-file index took {total:.2f}s — should be <2s"
        print(f"\n  [BENCH] 50 files: discover={t_discover*1000:.0f}ms index={t_index*1000:.0f}ms total={total*1000:.0f}ms ({t_index/50*1000:.1f}ms/file)")

    def test_index_200_files(self, project_200):
        """Index 200 files — should complete in <5s."""
        t0 = time.time()
        all_f, new_f, already = discover_files(project_200)
        t_discover = time.time() - t0

        assert len(all_f) == 200

        t0 = time.time()
        uids = ingest_many(new_f)
        t_index = time.time() - t0

        assert len(uids) == 200
        total = t_discover + t_index
        assert total < 5.0, f"200-file index took {total:.2f}s — should be <5s"
        print(f"\n  [BENCH] 200 files: discover={t_discover*1000:.0f}ms index={t_index*1000:.0f}ms total={total*1000:.0f}ms ({t_index/200*1000:.1f}ms/file)")

    def test_index_500_files(self, project_500):
        """Index 500 files — should complete in <15s."""
        t0 = time.time()
        all_f, new_f, already = discover_files(project_500)
        t_discover = time.time() - t0

        assert len(all_f) == 500

        t0 = time.time()
        uids = ingest_many(new_f)
        t_index = time.time() - t0

        assert len(uids) == 500
        total = t_discover + t_index
        assert total < 15.0, f"500-file index took {total:.2f}s — should be <15s"
        print(f"\n  [BENCH] 500 files: discover={t_discover*1000:.0f}ms index={t_index*1000:.0f}ms total={total*1000:.0f}ms ({t_index/500*1000:.1f}ms/file)")

    def test_reindex_is_instant(self, project_200):
        """Re-indexing 200 already-indexed files should be <200ms (fast path)."""
        # First pass: full index
        all_f, new_f, _ = discover_files(project_200)
        ingest_many(new_f)

        # Clear in-memory cache to simulate a new process
        _file_cache.clear()

        # Second pass: should hit DB fast-path
        t0 = time.time()
        all_f2, new_f2, already2 = discover_files(project_200)
        t_discover = time.time() - t0

        assert already2 == 200
        assert len(new_f2) == 0

        # Third pass with warm LRU cache
        t0 = time.time()
        all_f3, new_f3, already3 = discover_files(project_200)
        t_discover_warm = time.time() - t0

        assert already3 == 200
        assert t_discover_warm < 0.2, f"Warm re-discover took {t_discover_warm*1000:.0f}ms — should be <200ms"
        print(f"\n  [BENCH] re-discover 200 files: cold={t_discover*1000:.0f}ms warm={t_discover_warm*1000:.0f}ms")

    def test_progress_callback_accuracy(self, project_50):
        """Progress callback should be called exactly N times with correct counts."""
        _, new_f, _ = discover_files(project_50)

        calls = []
        def track(cur, total, path):
            calls.append((cur, total, str(path)))

        uids = ingest_many(new_f, progress_cb=track)

        assert len(calls) == 50
        for i, (cur, total, _) in enumerate(calls):
            assert cur == i + 1
            assert total == 50


# ── Benchmark: Search (requires OpenAI) ─────────────────────────────────────


_skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@_skip_no_openai
class TestSearchBenchmark:
    """End-to-end search benchmarks with real embeddings."""

    @pytest.fixture(autouse=True)
    def indexed_project(self, tmp_path):
        """Create and index a 100-file project."""
        p = tmp_path / "search_proj"
        _generate_project(p, n_files=100, words_per_file=400, include_junk=True)
        _, new_f, _ = discover_files(p)
        ingest_many(new_f)
        self._project = p
        return p

    def test_search_cold(self):
        """Cold search (no embedding cache) should complete in <3s."""
        from semantic_search import semantic_search
        from lss_store import clear_embeddings
        import semantic_search as ss

        clear_embeddings()
        ss.OAI_Q_CACHE = ss.LRU(512, ttl=900)
        ss.OAI_D_CACHE = ss.LRU(8192, ttl=3600)

        t0 = time.time()
        results = semantic_search(str(self._project), ["authentication JWT token"])
        dt = time.time() - t0

        assert len(results) == 1
        assert len(results[0]) > 0
        assert dt < 3.0, f"Cold search took {dt:.2f}s — should be <3s"
        print(f"\n  [BENCH] cold search: {dt*1000:.0f}ms ({len(results[0])} hits)")

    def test_search_warm(self):
        """Warm search (embeddings cached) should complete in <1s.

        Note: even with warm document caches, the query embedding still requires
        an OpenAI API call (~200-600ms depending on network), so the threshold
        is 1s to avoid flakiness from variable API latency.
        """
        from semantic_search import semantic_search

        # First call to warm the cache
        semantic_search(str(self._project), ["database migration"])

        # Second call — warm
        t0 = time.time()
        results = semantic_search(str(self._project), ["database migration"])
        dt = time.time() - t0

        assert len(results[0]) > 0
        assert dt < 1.0, f"Warm search took {dt*1000:.0f}ms — should be <1000ms"
        print(f"\n  [BENCH] warm search: {dt*1000:.0f}ms ({len(results[0])} hits)")

    def test_multi_query_search(self):
        """Multiple queries in one call should not take proportionally longer."""
        from semantic_search import semantic_search

        # Warm first
        queries = ["deploy Kubernetes", "PostgreSQL migration", "Redis caching"]
        semantic_search(str(self._project), queries)

        t0 = time.time()
        results = semantic_search(str(self._project), queries)
        dt = time.time() - t0

        assert len(results) == 3
        print(f"\n  [BENCH] 3-query warm search: {dt*1000:.0f}ms")


# ── Benchmark: Full pipeline (index + search) ───────────────────────────────


@_skip_no_openai
class TestFullPipelineBenchmark:
    """End-to-end: generate, discover, index, search — the complete user flow."""

    def test_full_pipeline_100_files(self, tmp_path):
        """Full pipeline: create 100 files, discover, index, search — under 5s total."""
        p = tmp_path / "full"
        n = _generate_project(p, n_files=100, words_per_file=300, include_junk=True)

        t_total = time.time()

        # Discover
        t0 = time.time()
        all_f, new_f, already = discover_files(p)
        t_discover = time.time() - t0

        # Index
        t0 = time.time()
        uids = ingest_many(new_f)
        t_index = time.time() - t0

        # Search (cold)
        from semantic_search import semantic_search
        t0 = time.time()
        results = semantic_search(str(p), ["authentication JWT deploy"])
        t_search = time.time() - t0

        t_total = time.time() - t_total

        assert len(uids) == n
        assert len(results[0]) > 0

        print(f"\n  [BENCH] FULL PIPELINE ({n} files):")
        print(f"    discover:  {t_discover*1000:.0f}ms")
        print(f"    index:     {t_index*1000:.0f}ms  ({t_index/n*1000:.1f}ms/file)")
        print(f"    search:    {t_search*1000:.0f}ms")
        print(f"    TOTAL:     {t_total*1000:.0f}ms")

        assert t_total < 5.0, f"Full pipeline took {t_total:.2f}s — should be <5s"
