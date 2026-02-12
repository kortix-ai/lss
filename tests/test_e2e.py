"""End-to-end tests — exercises the full CLI + search + sync pipeline.

These tests run the actual ``lss`` CLI commands as a user would, verifying
the complete flow from indexing through search with real OpenAI embeddings.
The lss-sync daemon is tested via its Python API (DebouncedIndexer) to
avoid flaky timing issues with background processes.

All tests that call the OpenAI API are gated on OPENAI_API_KEY.
"""

import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pytest

from lss_cli import main as lss_main, __version__, _C
import lss_config


# Disable colors in tests for predictable output matching
@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run(argv, capture=True):
    """Run an lss CLI command, return (exit_code, stdout, stderr)."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out = StringIO()
    sys.stderr = err = StringIO()
    try:
        rc = lss_main(argv)
    except SystemExit as e:
        rc = e.code if e.code is not None else 0
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def content_dir(tmp_path):
    """Create a realistic project directory for e2e testing."""
    d = tmp_path / "project"
    d.mkdir()

    (d / "README.md").write_text(
        "# MyApp\n\n"
        "A web application built with FastAPI and PostgreSQL.\n"
        "Deployed on AWS EKS using Kubernetes and ArgoCD.\n"
        "CI/CD runs on GitHub Actions with automated testing.\n"
    )
    (d / "architecture.md").write_text(
        "## Architecture\n\n"
        "The backend uses FastAPI with SQLAlchemy ORM.\n"
        "Authentication via JWT tokens with RSA-256 signing.\n"
        "Redis handles caching and background job queues.\n"
        "WebSocket connections use Socket.io for real-time updates.\n"
    )
    (d / "notes.txt").write_text(
        "Meeting notes 2026-02-08:\n"
        "- Discussed migrating from MySQL to PostgreSQL\n"
        "- Team agreed on Alembic for database migrations\n"
        "- Need to update connection pooling config\n"
    )
    (d / "config.json").write_text(
        '{"app_name": "MyApp", "version": "2.1.0", '
        '"database": {"host": "db.internal", "port": 5432}}'
    )
    # Binary file (should be skipped)
    (d / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    # Nested directory
    sub = d / "docs"
    sub.mkdir()
    (sub / "api.md").write_text(
        "# API Reference\n\n"
        "## POST /auth/login\n"
        "Authenticates a user and returns a JWT token.\n"
        "Requires email and password in the request body.\n"
    )
    return d


# ── E2E: Full CLI pipeline (no OpenAI needed) ───────────────────────────────


class TestCLIPipeline:
    """Tests that exercise multiple CLI commands in sequence."""

    def test_full_lifecycle_no_search(self, content_dir):
        """version -> index -> ls -> sweep -> ls (empty)."""
        # version
        rc, out, _ = _run(["version"])
        assert rc == 0
        assert __version__ in out

        # db-path
        rc, out, _ = _run(["db-path"])
        assert rc == 0
        assert "lss.db" in out

        # index a file
        rc, out, _ = _run(["index", str(content_dir / "README.md")])
        assert rc == 0
        assert "indexed" in out.lower() or "Indexed" in out

        # ls should show the file
        rc, out, _ = _run(["ls"])
        assert rc == 0
        assert "README.md" in out or "readme" in out.lower()

        # sweep --clear-all
        rc, out, _ = _run(["sweep", "--clear-all"])
        assert rc == 0
        assert "clear" in out.lower() or "Clear" in out

        # ls should be empty now
        rc, out, _ = _run(["ls"])
        assert rc == 0
        assert "no files" in out.lower() or "No files" in out

    def test_index_nonexistent(self):
        """Indexing a missing file should fail gracefully."""
        rc, _, err = _run(["index", "/no/such/file.txt"])
        assert rc == 2
        assert "error" in err.lower() or "not found" in err.lower()

    def test_index_binary_file(self, content_dir):
        """Indexing a binary file should fail gracefully."""
        rc, _, err = _run(["index", str(content_dir / "logo.png")])
        assert rc == 2

    def test_index_directory(self, content_dir):
        """'lss index <dir>' should index all text files in the directory."""
        rc, out, _ = _run(["index", str(content_dir)])
        assert rc == 0
        assert "indexed" in out.lower() or "Indexed" in out


# ── E2E: Config management (watch / exclude / status) ───────────────────────


class TestConfigManagement:
    """Tests for watch, exclude, and status commands."""

    def test_watch_add_list_remove(self, tmp_path):
        """Full watch path lifecycle: add -> list -> remove."""
        test_dir = tmp_path / "watched"
        test_dir.mkdir()

        # List (empty)
        rc, out, _ = _run(["watch", "list"])
        assert rc == 0
        assert "no watched" in out.lower() or "No watched" in out

        # Add
        rc, out, _ = _run(["watch", "add", str(test_dir)])
        assert rc == 0
        assert "Added" in out

        # List (should show the path)
        rc, out, _ = _run(["watch", "list"])
        assert rc == 0
        assert str(test_dir.resolve()) in out

        # Add duplicate
        rc, out, _ = _run(["watch", "add", str(test_dir)])
        assert rc == 0
        assert "Already watching" in out

        # Remove
        rc, out, _ = _run(["watch", "remove", str(test_dir)])
        assert rc == 0
        assert "Removed" in out

        # Remove again (should fail)
        rc, out, _ = _run(["watch", "remove", str(test_dir)])
        assert rc == 1
        assert "Not watching" in out

    def test_exclude_add_list_remove(self):
        """Full exclusion pattern lifecycle: add -> list -> remove."""
        # List (empty)
        rc, out, _ = _run(["exclude", "list"])
        assert rc == 0

        # Add a glob pattern
        rc, out, _ = _run(["exclude", "add", "*.log"])
        assert rc == 0
        assert "Added" in out

        # Add a dir name
        rc, out, _ = _run(["exclude", "add", "node_modules"])
        assert rc == 0

        # List
        rc, out, _ = _run(["exclude", "list"])
        assert rc == 0
        assert "*.log" in out
        assert "node_modules" in out

        # Add duplicate
        rc, out, _ = _run(["exclude", "add", "*.log"])
        assert rc == 0
        assert "Already excluded" in out

        # Remove
        rc, out, _ = _run(["exclude", "remove", "*.log"])
        assert rc == 0
        assert "Removed" in out

        # Remove non-existent
        rc, out, _ = _run(["exclude", "remove", "*.xyz"])
        assert rc == 1
        assert "Not excluded" in out

    def test_status_shows_config(self, content_dir):
        """Status command should show watched paths, exclusions, and DB stats."""
        _run(["watch", "add", str(content_dir)])
        _run(["exclude", "add", "*.log"])
        _run(["index", str(content_dir / "README.md")])

        rc, out, _ = _run(["status"])
        assert rc == 0
        assert "lss" in out
        assert "data dir" in out.lower() or "data" in out.lower()
        assert str(content_dir.resolve()) in out
        assert "*.log" in out

    def test_config_persists_across_calls(self, tmp_path):
        """Config written by 'watch add' should be readable on next call."""
        test_dir = tmp_path / "persist_test"
        test_dir.mkdir()

        _run(["watch", "add", str(test_dir)])
        _run(["exclude", "add", "build"])

        cfg = lss_config.load_config()
        assert str(test_dir.resolve()) in cfg["watch_paths"]
        assert "build" in cfg["exclude_patterns"]


# ── E2E: Search pipeline (requires OpenAI) ──────────────────────────────────

_skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@_skip_no_openai
class TestSearchE2E:
    """Full search pipeline tests with real OpenAI embeddings.

    Uses a single shared content_dir pre-indexed once to avoid
    DB locking issues from repeated ingest_many calls.
    """

    @pytest.fixture(autouse=True)
    def indexed_dir(self, content_dir):
        """Index content_dir once and make it available to all tests."""
        from lss_store import ingest_many
        ingest_many(content_dir)
        self._content_dir = content_dir
        return content_dir

    def test_search_directory_human_output(self):
        """Search a directory with human-readable output."""
        rc, out, err = _run(["search", "database migration", "-p", str(self._content_dir),
                             "--no-index"])
        assert rc == 0, f"search failed: {err}"
        # Should have some output (file paths, snippets)
        assert len(out.strip()) > 0

    def test_search_directory_json_output(self):
        """Search a directory with JSON output."""
        rc, out, err = _run([
            "search", "authentication JWT", "-p", str(self._content_dir),
            "--json", "-k", "3", "--no-index"
        ])
        assert rc == 0, f"search failed (rc={rc}): stderr={err}, stdout={out[:200]}"
        payload = json.loads(out)
        assert isinstance(payload, list)
        assert len(payload) == 1
        assert payload[0]["query"] == "authentication JWT"
        hits = payload[0]["hits"]
        assert len(hits) > 0
        assert len(hits) <= 3
        for h in hits:
            assert "file_path" in h
            assert "score" in h
            assert "snippet" in h
            assert Path(h["file_path"]).exists()

    def test_search_single_file(self):
        """Search within a single file."""
        rc, out, err = _run([
            "search", "caching strategy", "-p", str(self._content_dir / "architecture.md"),
            "--json", "--no-index"
        ])
        assert rc == 0, f"search failed: {err}"
        payload = json.loads(out)
        assert len(payload) == 1

    def test_search_multiple_queries(self):
        """Multiple queries in one call should return results for each."""
        rc, out, err = _run([
            "search", "deployment pipeline", "database migrations",
            "-p", str(self._content_dir),
            "--json", "-k", "2", "--no-index"
        ])
        assert rc == 0, f"search failed: {err}"
        payload = json.loads(out)
        assert len(payload) == 2
        assert payload[0]["query"] == "deployment pipeline"
        assert payload[1]["query"] == "database migrations"

    def test_search_empty_directory(self, tmp_path):
        """Searching an empty directory should return 0 hits."""
        empty = tmp_path / "empty"
        empty.mkdir()
        rc, out, err = _run(["search", "anything", "-p", str(empty), "--json"])
        assert rc == 0, f"search failed: {err}"
        payload = json.loads(out)
        assert len(payload[0]["hits"]) == 0

    def test_search_respects_limit(self):
        """The -k flag should cap result count."""
        rc, out, err = _run([
            "search", "web application", "-p", str(self._content_dir),
            "--json", "-k", "1", "--no-index"
        ])
        assert rc == 0, f"search failed: {err}"
        payload = json.loads(out)
        assert len(payload[0]["hits"]) <= 1

    def test_index_then_search(self):
        """Explicitly index, then search with --no-index."""
        rc, out, err = _run([
            "search", "Kubernetes deployment", "-p", str(self._content_dir / "README.md"),
            "--json", "--no-index"
        ])
        assert rc == 0, f"search failed: {err}"
        payload = json.loads(out)
        assert len(payload[0]["hits"]) > 0

    def test_implicit_search_with_path(self):
        """'lss <query> <path>' without 'search' subcommand should work."""
        rc, out, err = _run([
            "database migration", str(self._content_dir), "--json", "--no-index"
        ])
        assert rc == 0, f"implicit search failed: {err}"
        payload = json.loads(out)
        assert len(payload) == 1

    def test_smart_path_detection(self):
        """Path at end of positional args should be auto-detected."""
        rc, out, err = _run([
            "search", "authentication", str(self._content_dir),
            "--json", "--no-index"
        ])
        assert rc == 0, f"smart path detection failed: {err}"
        payload = json.loads(out)
        assert len(payload) == 1
        assert payload[0]["query"] == "authentication"


# ── E2E: lss-sync DebouncedIndexer (no OpenAI needed) ───────────────────────


class TestSyncE2E:
    """Tests the file-watcher indexer pipeline using the Python API directly."""

    def test_sync_detects_new_file(self, tmp_path):
        """DebouncedIndexer should index a newly created file."""
        from lss_sync import DebouncedIndexer
        from lss_store import _init_db

        watch_dir = tmp_path / "watched"
        watch_dir.mkdir()

        con = _init_db()
        con.close()

        indexer = DebouncedIndexer(
            [str(watch_dir)], debounce=0.1
        )

        test_file = watch_dir / "test.md"
        test_file.write_text("This is a test file about machine learning and neural networks")
        indexer.file_changed(str(test_file))
        time.sleep(1.0)

        import sqlite3
        con = sqlite3.connect(str(lss_config.LSS_DB))
        rows = con.execute("SELECT path FROM files WHERE status='active'").fetchall()
        con.close()
        paths = [r[0] for r in rows]
        assert any("test.md" in p for p in paths), f"test.md not in indexed files: {paths}"

    def test_sync_detects_deletion(self, tmp_path):
        """DebouncedIndexer should remove a deleted file from the index."""
        from lss_sync import DebouncedIndexer
        from lss_store import ensure_indexed, _init_db

        watch_dir = tmp_path / "watched"
        watch_dir.mkdir()

        test_file = watch_dir / "deleteme.md"
        test_file.write_text("Temporary file for deletion test")
        ensure_indexed(test_file)

        import sqlite3
        con = sqlite3.connect(str(lss_config.LSS_DB))
        count = con.execute("SELECT COUNT(*) FROM files WHERE status='active'").fetchone()[0]
        con.close()
        assert count >= 1

        test_file.unlink()
        indexer = DebouncedIndexer([str(watch_dir)], debounce=0.1)
        indexer.file_deleted(str(test_file))
        time.sleep(1.0)

        con = sqlite3.connect(str(lss_config.LSS_DB))
        rows = con.execute("SELECT path FROM files WHERE status='active'").fetchall()
        con.close()
        paths = [r[0] for r in rows]
        assert not any("deleteme.md" in p for p in paths), f"deleteme.md still indexed: {paths}"

    def test_sync_excludes_patterns(self):
        """LSSSyncHandler should respect exclusion patterns."""
        from lss_sync import LSSSyncHandler, DebouncedIndexer

        indexer = DebouncedIndexer(["/tmp/test"], exclude_patterns=["*.log", "build"])
        handler = LSSSyncHandler(indexer, exclude_patterns=["*.log", "build"])

        assert handler._should_ignore("/tmp/test/app.log") is True
        assert handler._should_ignore("/tmp/test/build/output.js") is True
        assert handler._should_ignore("/tmp/test/readme.md") is False

    def test_sync_ignores_chromium_cache_and_indexes_real_file(self, tmp_path):
        """Watcher should ignore Chromium cache churn while indexing real files."""
        from lss_sync import DebouncedIndexer

        watch_dir = tmp_path / "workspace"
        watch_dir.mkdir()

        real_file = watch_dir / "notes.md"
        real_file.write_text("Deployment notes for Kubernetes and JWT auth")

        cache_file = watch_dir / ".cache" / "chromium" / "Default" / "Cache" / "Cache_Data" / "d63b2f046ee63888_0"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(b"\x00\x01\x02\x03" * 64)

        indexer = DebouncedIndexer([str(watch_dir)], debounce=0.1)
        indexer.file_changed(str(cache_file))
        indexer.file_changed(str(real_file))
        time.sleep(1.0)

        import sqlite3
        con = sqlite3.connect(str(lss_config.LSS_DB))
        rows = con.execute("SELECT path FROM files WHERE status='active'").fetchall()
        con.close()
        paths = [r[0] for r in rows]

        assert any("notes.md" in p for p in paths)
        assert not any("Cache_Data" in p for p in paths)


# ── E2E: Confirmation + progress (no OpenAI needed) ─────────────────────────


class TestConfirmationAndProgress:
    """Tests for the interactive confirmation and progress reporting."""

    def test_index_dir_yes_flag(self, content_dir):
        """'lss index <dir> --yes' should skip confirmation and index."""
        rc, out, err = _run(["index", str(content_dir), "--yes"])
        assert rc == 0
        assert "indexed" in out.lower() or "Indexed" in out

    def test_index_dir_already_indexed(self, content_dir):
        """Re-indexing an already-indexed dir should say 'already indexed'."""
        # First index
        _run(["index", str(content_dir)])
        # Second index — should detect all files are already indexed
        rc, out, _ = _run(["index", str(content_dir)])
        assert rc == 0
        assert "already indexed" in out.lower()

    def test_index_noninteractive_auto_confirms(self, content_dir):
        """Non-interactive (piped stdout) should auto-confirm silently."""
        # _run() uses StringIO which is non-TTY → auto-confirms
        rc, out, _ = _run(["index", str(content_dir)])
        assert rc == 0
        assert "indexed" in out.lower() or "Indexed" in out

    def test_discover_files_counts(self, content_dir):
        """discover_files should return correct counts."""
        from lss_store import discover_files

        all_files, new_files, already = discover_files(content_dir)
        # content_dir has 5 text files (README.md, architecture.md, notes.txt, config.json, docs/api.md)
        # logo.png is binary → excluded
        assert len(all_files) == 5
        assert len(new_files) == 5
        assert already == 0  # nothing indexed yet

        # Index one file, then re-check
        from lss_store import ensure_indexed
        ensure_indexed(content_dir / "README.md")

        all_files2, new_files2, already2 = discover_files(content_dir)
        assert len(all_files2) == 5
        assert len(new_files2) == 4  # one less new
        assert already2 == 1

    def test_progress_callback_called(self, content_dir):
        """ingest_many with progress_cb should call it for each file."""
        from lss_store import ingest_many

        calls = []
        def track(cur, total, path):
            calls.append((cur, total, str(path)))

        uids = ingest_many(content_dir, progress_cb=track)
        assert len(calls) == len(uids)
        # Each call should have incrementing current
        for i, (cur, total, _) in enumerate(calls):
            assert cur == i + 1
            assert total == len(uids)
