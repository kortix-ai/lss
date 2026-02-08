"""Tests for query-time search filters: --ext, --exclude-ext, --exclude-pattern.

These filters allow narrowing search results at query time without re-indexing.
Extension filters are applied in SQL (efficient, pre-scoring).
Content regex exclusion is applied post-scoring.
"""

import json
import os
import re
import sys
from io import StringIO
from pathlib import Path

import pytest

import lss_config
import lss_store
from lss_cli import main as lss_main, _C


# Disable colors in tests
@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)


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
def mixed_project(tmp_path):
    """Create a project with multiple file types for filter testing."""
    d = tmp_path / "project"
    d.mkdir()

    (d / "auth.py").write_text(
        "def authenticate(user, password):\n"
        "    \"\"\"Authenticate user with JWT token.\"\"\"\n"
        "    token = create_jwt(user)\n"
        "    return token\n"
    )
    (d / "auth.ts").write_text(
        "export function authenticate(user: string, password: string): string {\n"
        "  // Authenticate user with JWT token\n"
        "  const token = createJWT(user);\n"
        "  return token;\n"
        "}\n"
    )
    (d / "config.yaml").write_text(
        "auth:\n"
        "  jwt_secret: supersecret\n"
        "  token_expiry: 3600\n"
        "  algorithm: RS256\n"
    )
    (d / "README.md").write_text(
        "# Auth Service\n\n"
        "Authentication service using JWT tokens.\n"
        "Supports OAuth2 and SAML authentication flows.\n"
    )
    (d / "deploy.sh").write_text(
        "#!/bin/bash\n"
        "# Deploy auth service to production\n"
        "docker build -t auth-service .\n"
        "kubectl apply -f k8s/\n"
    )
    (d / "test_auth.py").write_text(
        "def test_login():\n"
        "    \"\"\"Test user authentication endpoint.\"\"\"\n"
        "    response = client.post('/auth/login', json={'user': 'test', 'pass': 'test'})\n"
        "    assert response.status_code == 200\n"
        "    assert 'token' in response.json()\n"
    )
    (d / "schema.sql").write_text(
        "CREATE TABLE users (\n"
        "    id SERIAL PRIMARY KEY,\n"
        "    email TEXT NOT NULL UNIQUE,\n"
        "    password_hash TEXT NOT NULL,\n"
        "    created_at TIMESTAMP DEFAULT NOW()\n"
        ");\n"
    )
    (d / "errors.log").write_text(
        "2026-02-08 10:15:32 ERROR auth.authenticate: Invalid token for user john@example.com\n"
        "2026-02-08 10:16:01 ERROR auth.authenticate: Token expired for user alice@example.com\n"
    )
    return d


# ── semantic_search() filter parameter tests ──────────────────────────────


class TestExtensionFilterSemantic:
    """Test extension filters passed to semantic_search()."""

    def test_ext_filter_python_only(self, mixed_project):
        """--ext .py should only return results from .py files."""
        from semantic_search import semantic_search

        # Index first
        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["authenticate"],
            ext_include=[".py"],
        )
        assert len(results) == 1  # one query
        hits = results[0]
        assert len(hits) > 0
        for hit in hits:
            assert hit["file_path"].endswith(".py"), f"Expected .py, got {hit['file_path']}"

    def test_ext_filter_multiple(self, mixed_project):
        """--ext .py --ext .ts should return results from both."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["authenticate"],
            ext_include=[".py", ".ts"],
        )
        hits = results[0]
        assert len(hits) > 0
        extensions = {Path(h["file_path"]).suffix for h in hits}
        # Should only have .py and/or .ts
        assert extensions <= {".py", ".ts"}

    def test_exclude_ext_filter(self, mixed_project):
        """--exclude-ext .yaml should exclude yaml files."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["jwt"],
            ext_exclude=[".yaml"],
        )
        hits = results[0]
        for hit in hits:
            assert not hit["file_path"].endswith(".yaml"), f"Should exclude .yaml: {hit['file_path']}"

    def test_exclude_ext_multiple(self, mixed_project):
        """--exclude-ext .py --exclude-ext .ts should exclude both."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["authenticate"],
            ext_exclude=[".py", ".ts"],
        )
        hits = results[0]
        for hit in hits:
            ext = Path(hit["file_path"]).suffix
            assert ext not in (".py", ".ts"), f"Should exclude .py/.ts: {hit['file_path']}"

    def test_ext_include_and_exclude_combined(self, mixed_project):
        """--ext .py --exclude-ext should work, with include taking priority."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        # Include .py and .ts, but exclude .ts => only .py
        results = semantic_search(
            str(mixed_project), ["authenticate"],
            ext_include=[".py", ".ts"],
            ext_exclude=[".ts"],
        )
        hits = results[0]
        assert len(hits) > 0
        for hit in hits:
            assert hit["file_path"].endswith(".py")

    def test_ext_filter_no_match(self, mixed_project):
        """Extension filter with no matching files returns empty."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["authenticate"],
            ext_include=[".rs"],  # no Rust files
        )
        hits = results[0]
        assert len(hits) == 0

    def test_ext_filter_normalizes_dot(self, mixed_project):
        """Extension filter should work with or without leading dot."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        # Without dot
        results = semantic_search(
            str(mixed_project), ["authenticate"],
            ext_include=["py"],
        )
        hits = results[0]
        assert len(hits) > 0
        for hit in hits:
            assert hit["file_path"].endswith(".py")


class TestContentExcludePattern:
    """Test --exclude-pattern regex content filter."""

    def test_exclude_pattern_basic(self, mixed_project):
        """--exclude-pattern should filter out matching chunks."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        # Search for auth, but exclude results containing "test"
        results = semantic_search(
            str(mixed_project), ["authenticate"],
            exclude_patterns=[r"test_"],
        )
        hits = results[0]
        for hit in hits:
            snippet = hit.get("snippet", "")
            # The file path should not contain test_ either
            # (we filter on snippet content)
            assert "test_" not in snippet.lower() or "test_" not in hit["file_path"]

    def test_exclude_pattern_regex(self, mixed_project):
        """--exclude-pattern supports full regex."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        # Exclude lines with timestamps (like log entries)
        results = semantic_search(
            str(mixed_project), ["auth error"],
            exclude_patterns=[r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"],
        )
        hits = results[0]
        for hit in hits:
            snippet = hit.get("snippet", "")
            assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", snippet), \
                f"Should exclude timestamp pattern: {snippet[:80]}"

    def test_exclude_pattern_multiple(self, mixed_project):
        """Multiple --exclude-pattern flags combine with OR."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["auth"],
            exclude_patterns=[r"test_", r"#!/bin/bash"],
        )
        hits = results[0]
        for hit in hits:
            snippet = hit.get("snippet", "")
            assert "test_" not in snippet
            assert "#!/bin/bash" not in snippet

    def test_no_filters_returns_all(self, mixed_project):
        """Without filters, results include all file types."""
        from semantic_search import semantic_search

        all_files, new_files, _ = lss_store.discover_files(mixed_project)
        lss_store.ingest_many(new_files)

        results = semantic_search(
            str(mixed_project), ["jwt authentication"],
        )
        hits = results[0]
        assert len(hits) > 0
        # Should have results from multiple file types
        extensions = {Path(h["file_path"]).suffix for h in hits}
        assert len(extensions) > 1, f"Expected multiple extensions, got {extensions}"


# ── CLI integration tests ─────────────────────────────────────────────────


class TestFilterCLIArgs:
    """Test that filter CLI args are parsed and passed correctly."""

    def test_ext_flag_parsing(self, mixed_project):
        """lss 'query' -e .py should parse correctly."""
        rc, out, err = _run([
            "search", "authenticate",
            "-p", str(mixed_project),
            "-e", ".py",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        for hit in hits:
            assert hit["file_path"].endswith(".py")

    def test_ext_flag_multiple(self, mixed_project):
        """lss 'query' -e .py -e .ts should include both."""
        rc, out, err = _run([
            "search", "authenticate",
            "-p", str(mixed_project),
            "-e", ".py", "-e", ".ts",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        extensions = {Path(h["file_path"]).suffix for h in hits}
        assert extensions <= {".py", ".ts"}

    def test_exclude_ext_flag(self, mixed_project):
        """lss 'query' -E .yaml should exclude yaml."""
        rc, out, err = _run([
            "search", "jwt",
            "-p", str(mixed_project),
            "-E", ".yaml",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        for hit in hits:
            assert not hit["file_path"].endswith(".yaml")

    def test_exclude_pattern_flag(self, mixed_project):
        """lss 'query' -x 'pattern' should exclude matching snippets."""
        rc, out, err = _run([
            "search", "auth",
            "-p", str(mixed_project),
            "-x", r"test_",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        for hit in hits:
            assert "test_" not in hit.get("snippet", "")

    def test_smart_route_with_filters(self, mixed_project):
        """lss 'query' -e .py (without 'search' subcommand) should work."""
        rc, out, err = _run([
            "authenticate",
            "-p", str(mixed_project),
            "-e", ".py",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        for hit in hits:
            assert hit["file_path"].endswith(".py")

    def test_ext_without_dot(self, mixed_project):
        """lss 'query' -e py (without leading dot) should work."""
        rc, out, err = _run([
            "search", "authenticate",
            "-p", str(mixed_project),
            "-e", "py",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        for hit in hits:
            assert hit["file_path"].endswith(".py")

    def test_all_filters_combined(self, mixed_project):
        """All filter flags used together."""
        rc, out, err = _run([
            "search", "auth",
            "-p", str(mixed_project),
            "-e", ".py", "-e", ".ts",
            "-E", ".ts",
            "-x", r"test_",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        hits = data[0]["hits"]
        for hit in hits:
            # Include .py and .ts, but exclude .ts => only .py
            assert hit["file_path"].endswith(".py")
            assert "test_" not in hit.get("snippet", "")


# ── Edge cases ────────────────────────────────────────────────────────────


class TestFilterEdgeCases:
    """Edge cases for search filters."""

    def test_invalid_regex_pattern(self, mixed_project):
        """Invalid regex in --exclude-pattern should error gracefully."""
        rc, out, err = _run([
            "search", "auth",
            "-p", str(mixed_project),
            "-x", r"[invalid",
            "--json", "--yes",
        ])
        # Should either error with helpful message or treat as literal
        assert rc != 0 or "error" in err.lower()

    def test_empty_ext_filter(self, mixed_project):
        """Empty -e with no value should be handled."""
        # argparse should handle this — it requires a value
        rc, out, err = _run([
            "search", "auth",
            "-p", str(mixed_project),
            "-e",
        ])
        assert rc != 0  # argparse error

    def test_filters_with_file_scope(self, mixed_project):
        """Filters should still work when searching a single file."""
        rc, out, err = _run([
            "search", "authenticate",
            "-p", str(mixed_project / "auth.py"),
            "-e", ".py",
            "--json", "--yes",
        ])
        assert rc == 0
        data = json.loads(out)
        # Single file scope + matching extension = should work
        hits = data[0]["hits"]
        # auth.py is .py so it should match
        for hit in hits:
            assert hit["file_path"].endswith(".py")

    def test_filters_with_human_output(self, mixed_project):
        """Filters should work with non-JSON output too."""
        rc, out, err = _run([
            "search", "authenticate",
            "-p", str(mixed_project),
            "-e", ".py",
            "--yes",
        ])
        assert rc == 0
        # Should have output with .py files only
        # In human mode, file paths appear in output
        if out.strip():
            assert ".py" in out
