"""Tests for semantic_search — requires a real OpenAI API key."""

import json
import sqlite3
import sys
from io import StringIO
from pathlib import Path

import pytest

import lss_store
from lss_store import ingest_many, get_db_path
from semantic_search import semantic_search


# All tests in this module require the OpenAI API key.
pytestmark = pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


# ── Semantic search ──────────────────────────────────────────────────────────


def test_semantic_search_finds_related(sample_dir):
    """Searching for deployment-related terms should surface readme.md."""
    ingest_many(sample_dir)
    results = semantic_search(
        str(sample_dir), ["container deployment pipelines"], db_path=get_db_path()
    )

    assert len(results) == 1  # one result list per query
    hits = results[0]
    assert len(hits) > 0

    hit_paths = [h["file_path"] for h in hits]
    assert any("readme.md" in p for p in hit_paths), (
        f"Expected readme.md in hits, got: {hit_paths}"
    )


def test_bm25_keyword_match(sample_dir):
    """BM25 keyword search for 'JWT token RSA' should find notes.txt."""
    ingest_many(sample_dir)
    results = semantic_search(
        str(sample_dir), ["JWT token RSA"], db_path=get_db_path()
    )

    assert len(results) == 1
    hits = results[0]
    assert len(hits) > 0

    hit_paths = [h["file_path"] for h in hits]
    assert any("notes.txt" in p for p in hit_paths), (
        f"Expected notes.txt in hits, got: {hit_paths}"
    )


def test_search_returns_file_path(sample_dir):
    """Every hit must contain a file_path key pointing to a real file."""
    ingest_many(sample_dir)
    results = semantic_search(
        str(sample_dir), ["container deployment"], db_path=get_db_path()
    )
    for hits in results:
        for h in hits:
            assert "file_path" in h
            assert Path(h["file_path"]).exists(), (
                f"file_path does not exist: {h['file_path']}"
            )


def test_search_json_output(sample_dir):
    """CLI --json output should be valid JSON with query/hits structure."""
    from lss_cli import main

    ingest_many(sample_dir)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        rc = main(["search", str(sample_dir), "deployment", "--json"])
    finally:
        sys.stdout = old_stdout

    assert rc == 0
    output = captured.getvalue()
    payload = json.loads(output)

    assert isinstance(payload, list)
    assert len(payload) >= 1
    assert "query" in payload[0]
    assert "hits" in payload[0]


def test_search_empty_directory(tmp_path):
    """Searching an empty directory should return 0 hits and not crash."""
    empty = tmp_path / "empty"
    empty.mkdir()

    results = semantic_search(str(empty), ["anything at all"], db_path=get_db_path())
    assert len(results) == 1
    assert len(results[0]) == 0


def test_search_respects_limit(sample_dir):
    """The CLI -k flag should cap the number of hits per query."""
    from lss_cli import main

    ingest_many(sample_dir)

    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        rc = main(["search", str(sample_dir), "deployment", "-k", "2", "--json"])
    finally:
        sys.stdout = old_stdout

    assert rc == 0
    payload = json.loads(captured.getvalue())
    for entry in payload:
        assert len(entry["hits"]) <= 2
