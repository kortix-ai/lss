"""Tests for lss_store — the storage / indexing layer."""

import sqlite3
import time
from pathlib import Path

import pytest

import lss_store
from lss_store import (
    _init_db,
    _is_text_file,
    _span_chunk,
    _extract_text,
    ensure_indexed,
    get_db_path,
    get_file_uid,
    ingest_many,
    remove_files,
    sweep,
    clear_all,
)


# ── Schema ──────────────────────────────────────────────────────────────────


def test_init_db_creates_tables():
    """_init_db() should create the files, fts, and embeddings tables."""
    con = _init_db()
    cur = con.cursor()
    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    con.close()

    assert "files" in tables
    assert "fts" in tables
    assert "embeddings" in tables


# ── ensure_indexed ──────────────────────────────────────────────────────────


def test_index_text_file(sample_dir):
    """ensure_indexed on a text file returns a file_uid starting with 'f_'."""
    uid = ensure_indexed(sample_dir / "readme.md")
    assert uid.startswith("f_")

    # Verify the file appears in the files table
    con = sqlite3.connect(get_db_path())
    row = con.execute(
        "SELECT file_uid, status FROM files WHERE file_uid = ?", (uid,)
    ).fetchone()
    con.close()
    assert row is not None
    assert row[1] == "active"


def test_index_returns_same_uid_twice(sample_dir):
    """Indexing the same unmodified file twice returns the same uid."""
    uid1 = ensure_indexed(sample_dir / "readme.md")
    uid2 = ensure_indexed(sample_dir / "readme.md")
    assert uid1 == uid2


def test_index_detects_content_change(sample_dir):
    """After modifying content, force_reindex=True yields a new uid."""
    path = sample_dir / "readme.md"
    uid_before = ensure_indexed(path)

    # Modify the file content
    path.write_text("Completely new content about quantum computing")
    # Touch the mtime forward so the fast-path cache key changes
    new_mtime = time.time() + 10
    import os

    os.utime(path, (new_mtime, new_mtime))

    # Clear the in-memory cache so the change is visible
    lss_store._file_cache.clear()

    uid_after = ensure_indexed(path, force_reindex=True)
    assert uid_before != uid_after


# ── _is_text_file ───────────────────────────────────────────────────────────


def test_is_text_file_binary(sample_dir):
    """binary.png should be detected as non-text."""
    assert _is_text_file(sample_dir / "binary.png") is False


def test_is_text_file_text(sample_dir):
    """readme.md should be detected as text."""
    assert _is_text_file(sample_dir / "readme.md") is True


# ── ingest_many ─────────────────────────────────────────────────────────────


def test_ingest_many_directory(sample_dir):
    """ingest_many on a directory should index all text files (including nested)."""
    uids = ingest_many(sample_dir)
    # Expected text files: readme.md, notes.txt, code.py, data.json,
    # deep/nested/file.md, .hidden_file  (NOT binary.png, NOT node_modules/*)
    assert len(uids) >= 5  # at least the main 5; .hidden_file may also be indexed


def test_ingest_many_excludes_node_modules(sample_dir):
    """Files inside node_modules/ must not appear in the index."""
    ingest_many(sample_dir)

    con = sqlite3.connect(get_db_path())
    rows = con.execute("SELECT path FROM files").fetchall()
    con.close()

    paths = [r[0] for r in rows]
    for p in paths:
        assert "node_modules" not in p, f"node_modules file was indexed: {p}"


# ── remove_files ────────────────────────────────────────────────────────────


def test_remove_files(sample_dir):
    """remove_files should delete a file from the index."""
    path = sample_dir / "readme.md"
    ensure_indexed(path)

    # Clear cache so get_file_uid actually hits the DB
    lss_store._file_cache.clear()
    assert get_file_uid(path) is not None

    remove_files([path])
    lss_store._file_cache.clear()
    assert get_file_uid(path) is None


# ── sweep ───────────────────────────────────────────────────────────────────


def test_sweep_marks_missing(sample_dir):
    """If a file is deleted from disk, sweep() should mark it as 'missing'."""
    path = sample_dir / "readme.md"
    uid = ensure_indexed(path)

    # Delete the file from disk
    path.unlink()

    sweep()

    con = sqlite3.connect(get_db_path())
    row = con.execute(
        "SELECT status FROM files WHERE file_uid = ?", (uid,)
    ).fetchone()
    con.close()

    assert row is not None
    assert row[0] == "missing"


# ── get_file_uid ────────────────────────────────────────────────────────────


def test_get_file_uid_unindexed(sample_dir):
    """get_file_uid for a file that hasn't been indexed returns None."""
    uid = get_file_uid(sample_dir / "readme.md")
    assert uid is None


# ── clear_all ───────────────────────────────────────────────────────────────


def test_clear_all(sample_dir):
    """clear_all drops all rows and recreates empty schema."""
    ingest_many(sample_dir)

    con = sqlite3.connect(get_db_path())
    count_before = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    con.close()
    assert count_before > 0

    clear_all()

    con = sqlite3.connect(get_db_path())
    count_after = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    con.close()
    assert count_after == 0


# ── _span_chunk ─────────────────────────────────────────────────────────────


def test_span_chunking():
    """_span_chunk should split long text into multiple overlapping spans."""
    # Generate a 500-word text
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)

    spans = _span_chunk(text, words_per_span=220, stride=200)
    assert len(spans) > 1

    for chunk_type, span_text in spans:
        assert chunk_type == "simple"
        span_words = span_text.split()
        assert len(span_words) <= 220


# ── _extract_text (JSON) ───────────────────────────────────────────────────


def test_extract_json(sample_dir):
    """_extract_text on a JSON file should return its string values."""
    text = _extract_text(sample_dir / "data.json")
    assert "lss" in text
    assert "Local semantic search engine" in text
