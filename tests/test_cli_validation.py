"""Comprehensive CLI validation tests.

Exercises EVERY command, subcommand, and flag variation to ensure nothing
crashes, returns unexpected exit codes, or produces malformed output.
These are NOT deep feature tests (those live in test_lss_cli.py, test_e2e.py,
test_embedding_provider.py) — they're lightweight smoke tests for the full
surface area of the CLI.
"""

import json
import os
import sys
from io import StringIO
from pathlib import Path

import pytest

import lss_config
from lss_cli import main, __version__, _C, build_parser, _KNOWN_SUBCOMMANDS


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)


def _run(argv):
    """Run CLI, capture stdout/stderr, never raise on SystemExit."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = out = StringIO()
    sys.stderr = err = StringIO()
    try:
        rc = main(argv)
    except SystemExit as e:
        rc = e.code if e.code is not None else 0
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def content_dir(tmp_path):
    """A small project directory with various file types."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "readme.md").write_text("# Project\nDeployment guide for Kubernetes.\n")
    (d / "main.py").write_text(
        'def main():\n    """Entry point."""\n    print("hello")\n'
    )
    (d / "config.yaml").write_text("server:\n  port: 8080\n  host: 0.0.0.0\n")
    (d / "data.json").write_text('{"key": "value", "items": [1, 2, 3]}')
    (d / "notes.txt").write_text("Meeting notes: discussed auth flow with JWT tokens.")
    (d / "style.css").write_text("body { font-family: sans-serif; }")
    (d / "app.js").write_text('console.log("hello");\n')
    # Nested
    sub = d / "src" / "lib"
    sub.mkdir(parents=True)
    (sub / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    # Binary — should be skipped
    (d / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    # Hidden — should be skipped
    (d / ".secret").write_text("password=hunter2")
    # node_modules — should be excluded
    nm = d / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}")
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NO ARGS / HELP
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoArgsAndHelp:
    """Running lss with no args or -h should print help and not crash."""

    def test_no_args_returns_0(self):
        rc, out, err = _run([])
        assert rc == 0

    def test_no_args_prints_usage(self):
        rc, out, err = _run([])
        assert "usage" in out.lower() or "lss" in out.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VERSION
# ═══════════════════════════════════════════════════════════════════════════════


class TestVersion:
    def test_version_subcommand(self):
        rc, out, _ = _run(["version"])
        assert rc == 0
        assert __version__ in out

    def test_version_flag(self):
        """'lss -v' should print version (argparse exits with 0)."""
        rc, out, _ = _run(["-v"])
        assert rc == 0
        assert __version__ in out

    def test_version_long_flag(self):
        rc, out, _ = _run(["--version"])
        assert rc == 0
        assert __version__ in out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STATUS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatus:
    def test_status_returns_0(self):
        rc, out, _ = _run(["status"])
        assert rc == 0

    def test_status_shows_version(self):
        rc, out, _ = _run(["status"])
        assert __version__ in out

    def test_status_shows_provider(self):
        rc, out, _ = _run(["status"])
        assert "provider" in out.lower()

    def test_status_shows_model(self):
        rc, out, _ = _run(["status"])
        assert "model" in out.lower()

    def test_status_shows_dimensions(self):
        rc, out, _ = _run(["status"])
        assert "dimensions" in out.lower() or "dim" in out.lower()

    def test_status_shows_data_dir(self):
        rc, out, _ = _run(["status"])
        assert "data dir" in out.lower() or "lss" in out.lower()

    def test_status_debug(self):
        rc, out, _ = _run(["status", "--debug"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_config_show(self):
        rc, out, _ = _run(["config", "show"])
        assert rc == 0
        assert "provider" in out.lower()
        assert "model" in out.lower()

    def test_config_provider_invalid(self):
        rc, out, err = _run(["config", "provider", "bogus"])
        assert rc == 2
        assert "unknown" in err.lower() or "error" in err.lower()

    def test_config_provider_local(self):
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")
        rc, out, _ = _run(["config", "provider", "local"])
        assert rc == 0
        assert "local" in out.lower()

    def test_config_provider_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        rc, out, _ = _run(["config", "provider", "openai"])
        assert rc == 0

    def test_config_provider_switch_warns(self):
        """Switching provider should mention re-embedding."""
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")
        # Set to openai first
        cfg = lss_config.load_config()
        cfg["embedding_provider"] = "openai"
        lss_config.save_config(cfg)
        lss_config.EMBEDDING_PROVIDER = "openai"

        # Switch to local
        rc, out, _ = _run(["config", "provider", "local"])
        assert rc == 0
        assert "re-computed" in out.lower() or "note" in out.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. WATCH
# ═══════════════════════════════════════════════════════════════════════════════


class TestWatch:
    def test_watch_list_empty(self):
        rc, out, _ = _run(["watch", "list"])
        assert rc == 0

    def test_watch_add(self, tmp_path):
        d = tmp_path / "watched"
        d.mkdir()
        rc, out, _ = _run(["watch", "add", str(d)])
        assert rc == 0
        assert "Added" in out or "added" in out.lower()

    def test_watch_add_then_list(self, tmp_path):
        d = tmp_path / "watched"
        d.mkdir()
        _run(["watch", "add", str(d)])
        rc, out, _ = _run(["watch", "list"])
        assert rc == 0
        assert str(d.resolve()) in out

    def test_watch_add_duplicate(self, tmp_path):
        d = tmp_path / "watched"
        d.mkdir()
        _run(["watch", "add", str(d)])
        rc, out, _ = _run(["watch", "add", str(d)])
        assert rc == 0
        assert "already" in out.lower()

    def test_watch_remove(self, tmp_path):
        d = tmp_path / "watched"
        d.mkdir()
        _run(["watch", "add", str(d)])
        rc, out, _ = _run(["watch", "remove", str(d)])
        assert rc == 0
        assert "removed" in out.lower()

    def test_watch_remove_nonexistent(self, tmp_path):
        rc, out, _ = _run(["watch", "remove", "/no/such/path"])
        assert rc == 1

    def test_watch_add_debug(self, tmp_path):
        d = tmp_path / "watched"
        d.mkdir()
        rc, _, _ = _run(["watch", "add", str(d)])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INCLUDE
# ═══════════════════════════════════════════════════════════════════════════════


class TestInclude:
    def test_include_list_empty(self):
        rc, out, _ = _run(["include", "list"])
        assert rc == 0
        assert "built-in" in out.lower()
        assert "no custom" in out.lower()

    def test_include_add_new(self):
        rc, out, _ = _run(["include", "add", ".xyz"])
        assert rc == 0
        assert ".xyz" in out

    def test_include_add_without_dot(self):
        rc, out, _ = _run(["include", "add", "abc"])
        assert rc == 0
        assert ".abc" in out

    def test_include_add_uppercase_normalised(self):
        rc, out, _ = _run(["include", "add", ".XYZ"])
        assert rc == 0
        assert ".xyz" in out

    def test_include_add_duplicate(self):
        _run(["include", "add", ".qqq"])
        rc, out, _ = _run(["include", "add", ".qqq"])
        assert rc == 0
        assert "already" in out.lower()

    def test_include_add_builtin(self):
        rc, out, _ = _run(["include", "add", ".py"])
        assert rc == 0
        assert "already" in out.lower() or "built-in" in out.lower()

    def test_include_list_after_add(self):
        _run(["include", "add", ".zzz"])
        rc, out, _ = _run(["include", "list"])
        assert rc == 0
        assert ".zzz" in out

    def test_include_remove(self):
        _run(["include", "add", ".zzz"])
        rc, out, _ = _run(["include", "remove", ".zzz"])
        assert rc == 0
        assert "removed" in out.lower()

    def test_include_remove_not_found(self):
        rc, out, _ = _run(["include", "remove", ".nope"])
        assert rc == 1

    def test_include_persists(self):
        _run(["include", "add", ".custom123"])
        cfg = lss_config.load_config()
        assert ".custom123" in cfg.get("include_extensions", [])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXCLUDE
# ═══════════════════════════════════════════════════════════════════════════════


class TestExclude:
    def test_exclude_list_empty(self):
        rc, out, _ = _run(["exclude", "list"])
        assert rc == 0

    def test_exclude_add(self):
        rc, out, _ = _run(["exclude", "add", "*.log"])
        assert rc == 0
        assert "*.log" in out

    def test_exclude_list_after_add(self):
        _run(["exclude", "add", "*.tmp"])
        rc, out, _ = _run(["exclude", "list"])
        assert rc == 0
        assert "*.tmp" in out

    def test_exclude_add_duplicate(self):
        _run(["exclude", "add", "*.bak"])
        rc, out, _ = _run(["exclude", "add", "*.bak"])
        assert rc == 0
        assert "already" in out.lower()

    def test_exclude_remove(self):
        _run(["exclude", "add", "*.old"])
        rc, out, _ = _run(["exclude", "remove", "*.old"])
        assert rc == 0

    def test_exclude_remove_not_found(self):
        rc, out, _ = _run(["exclude", "remove", "*.nope"])
        assert rc == 1

    def test_exclude_add_dir_pattern(self):
        rc, out, _ = _run(["exclude", "add", "vendor"])
        assert rc == 0

    def test_exclude_add_path_pattern(self):
        rc, out, _ = _run(["exclude", "add", "src/generated"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. INDEX
# ═══════════════════════════════════════════════════════════════════════════════


class TestIndex:
    def test_index_single_file(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir / "readme.md")])
        assert rc == 0
        assert "indexed" in out.lower() or "Indexed" in out

    def test_index_nonexistent(self):
        rc, _, err = _run(["index", "/no/such/file.txt"])
        assert rc == 2
        assert "error" in err.lower() or "not found" in err.lower()

    def test_index_binary_file(self, content_dir):
        rc, _, err = _run(["index", str(content_dir / "logo.png")])
        assert rc == 2

    def test_index_quiet(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir / "readme.md"), "-q"])
        assert rc == 0
        # Quiet mode should produce less output
        # (it's OK if there's some output, just shouldn't crash)

    def test_index_directory_yes(self, content_dir):
        """'lss index <dir> -y' should index without asking."""
        rc, out, _ = _run(["index", str(content_dir), "-y"])
        assert rc == 0

    def test_index_directory_quiet_yes(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir), "-q", "-y"])
        assert rc == 0

    def test_index_with_debug(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir / "readme.md"), "--debug"])
        assert rc == 0

    def test_index_python_file(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir / "main.py")])
        assert rc == 0

    def test_index_json_file(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir / "data.json")])
        assert rc == 0

    def test_index_nested_file(self, content_dir):
        rc, out, _ = _run(["index", str(content_dir / "src" / "lib" / "utils.py")])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. LS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLs:
    def test_ls_empty_no_db(self):
        rc, out, _ = _run(["ls"])
        assert rc == 0
        assert "no files" in out.lower() or "No files" in out

    def test_ls_after_indexing(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["ls"])
        assert rc == 0
        assert "readme.md" in out
        assert "active" in out.lower()

    def test_ls_shows_count(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        _run(["index", str(content_dir / "main.py")])
        rc, out, _ = _run(["ls"])
        assert rc == 0
        assert "2" in out  # "2 files indexed"

    def test_ls_debug(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["ls", "--debug"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DB-PATH
# ═══════════════════════════════════════════════════════════════════════════════


class TestDbPath:
    def test_dbpath(self):
        rc, out, _ = _run(["db-path"])
        assert rc == 0
        assert "lss.db" in out

    def test_dbpath_debug(self):
        rc, out, _ = _run(["db-path", "--debug"])
        assert rc == 0
        assert "lss.db" in out


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SWEEP
# ═══════════════════════════════════════════════════════════════════════════════


class TestSweep:
    def test_sweep_default(self, content_dir):
        """Default sweep should succeed (even with empty DB)."""
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep"])
        assert rc == 0
        assert "sweep" in out.lower() or "complete" in out.lower()

    def test_sweep_clear_all(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep", "--clear-all"])
        assert rc == 0
        assert "clear" in out.lower()

    def test_sweep_clear_embeddings_all(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep", "--clear-embeddings", "0"])
        assert rc == 0

    def test_sweep_clear_embeddings_days(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep", "--clear-embeddings", "7"])
        assert rc == 0

    def test_sweep_remove_path(self, content_dir):
        f = content_dir / "readme.md"
        _run(["index", str(f)])
        rc, out, _ = _run(["sweep", "--remove", str(f)])
        assert rc == 0

    def test_sweep_remove_nonexistent(self):
        from lss_store import _init_db
        _init_db().close()
        rc, out, _ = _run(["sweep", "--remove", "/no/such/file.txt"])
        assert rc == 0
        assert "nothing" in out.lower()

    def test_sweep_no_optimize(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep", "--no-optimize"])
        assert rc == 0

    def test_sweep_retention_days(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep", "--retention-days", "7"])
        assert rc == 0

    def test_sweep_debug(self, content_dir):
        _run(["index", str(content_dir / "readme.md")])
        rc, out, _ = _run(["sweep", "--debug"])
        assert rc == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SEARCH (no API key required checks)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchNoApi:
    """Search command validation that doesn't require actual API calls."""

    def test_search_no_query(self):
        """'lss search' with no query should print usage hint."""
        rc, _, err = _run(["search"])
        assert rc == 2
        assert "error" in err.lower() or "no query" in err.lower()

    def test_search_nonexistent_path(self):
        rc, _, err = _run(["search", "hello", "-p", "/no/such/dir"])
        assert rc == 2
        assert "error" in err.lower() or "not found" in err.lower()

    def test_search_no_color_flag(self, monkeypatch):
        """--no-color should not crash."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "openai")
        rc, _, _ = _run(["search", "hello", "--no-color"])
        # Will fail due to no API key, but shouldn't crash (rc=1 is fine)
        assert rc in (1, 2)


class TestSearchWithProvider:
    """Search with local provider — exercises the full pipeline."""

    @pytest.fixture(autouse=True)
    def _local_provider(self, monkeypatch):
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "local")
        import semantic_search
        monkeypatch.setattr(semantic_search, "EMBED_PROVIDER", "local")
        monkeypatch.setattr(semantic_search, "EMBED_MODEL", lss_config.LOCAL_MODEL)
        monkeypatch.setattr(semantic_search, "EMBED_DIM", lss_config.LOCAL_DIM)

    def test_search_local_basic(self, content_dir):
        """Full search pipeline with local embeddings should return results."""
        # Index first
        _run(["index", str(content_dir), "-y"])
        rc, out, err = _run(["search", "deployment", "-p", str(content_dir)])
        assert rc == 0, f"stderr: {err}"

    def test_search_local_json(self, content_dir):
        _run(["index", str(content_dir), "-y"])
        rc, out, err = _run(["search", "kubernetes", "-p", str(content_dir), "--json"])
        assert rc == 0, f"stderr: {err}"
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "query" in data[0]
        assert "hits" in data[0]

    def test_search_local_limit(self, content_dir):
        _run(["index", str(content_dir), "-y"])
        rc, out, err = _run([
            "search", "python", "-p", str(content_dir), "--json", "-k", "2"
        ])
        assert rc == 0, f"stderr: {err}"
        data = json.loads(out)
        assert len(data[0]["hits"]) <= 2

    def test_search_local_multiple_queries(self, content_dir):
        _run(["index", str(content_dir), "-y"])
        rc, out, err = _run([
            "search", "deployment", "auth tokens", "-p", str(content_dir), "--json"
        ])
        assert rc == 0, f"stderr: {err}"
        data = json.loads(out)
        assert len(data) == 2

    def test_search_local_no_index_flag(self, content_dir):
        """--no-index should search without auto-indexing."""
        # Don't index first — should return empty
        rc, out, err = _run([
            "search", "deployment", "-p", str(content_dir), "--no-index", "--json"
        ])
        assert rc == 0, f"stderr: {err}"

    def test_search_local_yes_flag(self, content_dir):
        """'-y' should skip confirmation."""
        rc, out, err = _run([
            "search", "deployment", "-p", str(content_dir), "-y"
        ])
        assert rc == 0, f"stderr: {err}"

    def test_search_implicit_routing(self, content_dir):
        """'lss "query" <path>' without 'search' should work."""
        _run(["index", str(content_dir), "-y"])
        # Use _run which goes through main() with smart routing
        rc, out, err = _run(["deployment", str(content_dir)])
        assert rc == 0, f"stderr: {err}"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SMART ROUTING
# ═══════════════════════════════════════════════════════════════════════════════


class TestSmartRouting:
    """First arg != known subcommand should route to search."""

    def test_known_subcommands_complete(self):
        """Every subcommand in the parser should be in _KNOWN_SUBCOMMANDS."""
        parser = build_parser()
        # Get all subparser choices
        for action in parser._subparsers._actions:
            if hasattr(action, "_parser_class"):
                for choice in action.choices:
                    assert choice in _KNOWN_SUBCOMMANDS, (
                        f"Parser subcommand '{choice}' missing from _KNOWN_SUBCOMMANDS"
                    )

    def test_unknown_arg_routes_to_search(self, monkeypatch):
        """An unknown first arg like 'hello' should be treated as a search query."""
        # This will fail at the provider check, but it shouldn't crash or
        # say "unknown command"
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "openai")
        rc, _, err = _run(["hello"])
        # Should try to search (rc=1 from missing API key) not print help
        assert rc in (0, 1, 2)
        # Should NOT contain "usage" from the top-level parser
        assert "lss <command>" not in err


# ═══════════════════════════════════════════════════════════════════════════════
# 14. UPDATE (dry check — don't actually upgrade)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdate:
    def test_update_runs(self):
        """'lss update' should at least start (queries PyPI)."""
        rc, out, err = _run(["update"])
        # May succeed (0) or fail to reach PyPI (1), both are fine
        assert rc in (0, 1)
        assert __version__ in out


# ═══════════════════════════════════════════════════════════════════════════════
# 15. FULL LIFECYCLE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullLifecycle:
    """End-to-end: config → watch → include → index → ls → search → sweep."""

    @pytest.fixture(autouse=True)
    def _local_provider(self, monkeypatch):
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "local")
        import semantic_search
        monkeypatch.setattr(semantic_search, "EMBED_PROVIDER", "local")
        monkeypatch.setattr(semantic_search, "EMBED_MODEL", lss_config.LOCAL_MODEL)
        monkeypatch.setattr(semantic_search, "EMBED_DIM", lss_config.LOCAL_DIM)

    def test_full_lifecycle(self, content_dir, tmp_path):
        """Exercise the full command sequence a user would run."""

        # 1. Status (fresh)
        rc, out, _ = _run(["status"])
        assert rc == 0

        # 2. Config
        rc, out, _ = _run(["config", "show"])
        assert rc == 0

        # 3. Add watch path
        rc, out, _ = _run(["watch", "add", str(content_dir)])
        assert rc == 0

        # 4. Add custom extension
        rc, out, _ = _run(["include", "add", ".custom"])
        assert rc == 0

        # 5. Add exclusion
        rc, out, _ = _run(["exclude", "add", "*.tmp"])
        assert rc == 0

        # 6. Index
        rc, out, _ = _run(["index", str(content_dir), "-y"])
        assert rc == 0

        # 7. List indexed files
        rc, out, _ = _run(["ls"])
        assert rc == 0
        assert "readme.md" in out

        # 8. Search
        rc, out, err = _run(["search", "kubernetes", "-p", str(content_dir), "--json"])
        assert rc == 0, f"stderr: {err}"
        data = json.loads(out)
        assert len(data) >= 1

        # 9. DB path
        rc, out, _ = _run(["db-path"])
        assert rc == 0

        # 10. Status (with data)
        rc, out, _ = _run(["status"])
        assert rc == 0
        assert "indexed files" in out.lower() or "custom extensions" in out.lower()

        # 11. Sweep
        rc, out, _ = _run(["sweep"])
        assert rc == 0

        # 12. Cleanup — remove watch, include, exclude
        _run(["watch", "remove", str(content_dir)])
        _run(["include", "remove", ".custom"])
        _run(["exclude", "remove", "*.tmp"])

        # 13. Verify cleanup
        rc, out, _ = _run(["watch", "list"])
        assert rc == 0

        # 14. Clear all
        rc, out, _ = _run(["sweep", "--clear-all"])
        assert rc == 0

    def test_index_then_search_multiple_formats(self, content_dir):
        """Index various file types and search across them."""
        _run(["index", str(content_dir), "-y"])

        # Search for something in markdown
        rc, out, err = _run(["search", "kubernetes", "-p", str(content_dir), "--json"])
        assert rc == 0, f"stderr: {err}"
        data = json.loads(out)
        hits = data[0]["hits"]
        assert any("readme" in h["file_path"].lower() for h in hits)

        # Search for something in Python
        rc, out, err = _run(["search", "entry point", "-p", str(content_dir), "--json"])
        assert rc == 0, f"stderr: {err}"

        # Search for something in YAML
        rc, out, err = _run(["search", "server port", "-p", str(content_dir), "--json"])
        assert rc == 0, f"stderr: {err}"


# ═══════════════════════════════════════════════════════════════════════════════
# 16. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and unusual inputs that should not crash."""

    def test_index_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        rc, out, _ = _run(["index", str(d), "-y"])
        assert rc == 0

    def test_index_dir_only_binary(self, tmp_path):
        d = tmp_path / "binaries"
        d.mkdir()
        (d / "a.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
        (d / "b.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 50)
        rc, out, _ = _run(["index", str(d), "-y"])
        assert rc == 0

    def test_search_empty_string_query(self):
        """Empty query string should show usage hint, not crash."""
        rc, _, err = _run(["search", ""])
        assert rc == 2

    def test_include_add_empty_ext(self):
        """Adding empty extension (just '.') should not crash."""
        rc, out, _ = _run(["include", "add", "."])
        assert rc == 0  # normalised to "." — unusual but shouldn't crash

    def test_exclude_add_empty_pattern(self):
        """Adding empty-ish pattern should not crash."""
        rc, out, _ = _run(["exclude", "add", ""])
        assert rc == 0  # edge case but shouldn't crash

    def test_index_file_then_re_index(self, content_dir):
        """Re-indexing an already-indexed file should be idempotent."""
        f = content_dir / "readme.md"
        rc1, _, _ = _run(["index", str(f)])
        rc2, out2, _ = _run(["index", str(f)])
        assert rc1 == 0
        assert rc2 == 0

    def test_sweep_on_empty_db(self):
        """Sweep on a freshly initialised DB should not crash."""
        from lss_store import _init_db
        _init_db().close()
        rc, out, _ = _run(["sweep"])
        assert rc == 0

    def test_ls_on_empty_db(self):
        """ls on a freshly initialised DB should print empty message."""
        from lss_store import _init_db
        _init_db().close()
        rc, out, _ = _run(["ls"])
        assert rc == 0

    def test_status_on_fresh_install(self):
        """Status with no DB and no config should not crash."""
        rc, out, _ = _run(["status"])
        assert rc == 0

    def test_search_single_file_path(self, content_dir, monkeypatch):
        """Searching with -p pointing to a single file."""
        try:
            import fastembed  # noqa: F401
        except ImportError:
            pytest.skip("fastembed not installed")
        monkeypatch.setattr(lss_config, "EMBEDDING_PROVIDER", "local")
        import semantic_search
        monkeypatch.setattr(semantic_search, "EMBED_PROVIDER", "local")
        monkeypatch.setattr(semantic_search, "EMBED_MODEL", lss_config.LOCAL_MODEL)
        monkeypatch.setattr(semantic_search, "EMBED_DIM", lss_config.LOCAL_DIM)

        f = content_dir / "readme.md"
        rc, out, err = _run(["search", "kubernetes", "-p", str(f)])
        assert rc == 0, f"stderr: {err}"
