"""Tests for lss_store — the storage / indexing layer."""

import sqlite3
import os
import time
from pathlib import Path

import pytest

import lss_store
from lss_store import (
    _init_db,
    _is_text_file,
    _path_uid,
    _span_chunk,
    _extract_text,
    discover_files,
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
    """After modifying content, force_reindex=True updates the content_sig."""
    path = sample_dir / "readme.md"
    uid_before = ensure_indexed(path)

    # Read the original content_sig
    con = sqlite3.connect(get_db_path())
    sig_before = con.execute(
        "SELECT content_sig FROM files WHERE file_uid = ?", (uid_before,)
    ).fetchone()[0]
    con.close()

    # Modify the file content
    path.write_text("Completely new content about quantum computing")
    # Touch the mtime forward so the fast-path cache key changes
    new_mtime = time.time() + 10
    import os

    os.utime(path, (new_mtime, new_mtime))

    # Clear the in-memory cache so the change is visible
    lss_store._file_cache.clear()

    uid_after = ensure_indexed(path, force_reindex=True)

    # UID is path-based so stays the same; content_sig must change
    assert uid_before == uid_after

    con = sqlite3.connect(get_db_path())
    sig_after = con.execute(
        "SELECT content_sig FROM files WHERE file_uid = ?", (uid_after,)
    ).fetchone()[0]
    con.close()
    assert sig_before != sig_after


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


# ── Path-based UID (duplicate-content bug fix) ─────────────────────────────


def test_path_uid_differs_for_same_content(tmp_path):
    """Two files with identical content must get different file_uids."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("# same\ndef shared(): pass\n")
    b.write_text("# same\ndef shared(): pass\n")

    uid_a = _path_uid(a.resolve())
    uid_b = _path_uid(b.resolve())
    assert uid_a != uid_b


def test_path_uid_stable_for_same_path(tmp_path):
    """Calling _path_uid twice on the same resolved path returns the same uid."""
    f = tmp_path / "stable.py"
    f.write_text("content")
    uid1 = _path_uid(f.resolve())
    uid2 = _path_uid(f.resolve())
    assert uid1 == uid2


def test_duplicate_content_all_indexed(tmp_path):
    """Files with identical content must ALL appear in the DB after ingest."""
    d = tmp_path / "dup_project"
    d.mkdir()

    # Create 5 files with the exact same content
    names = [f"pkg{i}/__init__.py" for i in range(5)]
    for name in names:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# package init\n")

    uids = ingest_many(d)

    # Every file must produce a unique uid
    assert len(uids) == 5
    assert len(set(uids)) == 5

    # All 5 rows must be in the DB
    con = sqlite3.connect(get_db_path())
    rows = con.execute(
        "SELECT file_uid, path FROM files WHERE status = 'active'"
    ).fetchall()
    con.close()
    assert len(rows) == 5


def test_duplicate_content_persists_across_sessions(tmp_path):
    """After indexing duplicate-content files, a fresh cache still finds them."""
    d = tmp_path / "dup_session"
    d.mkdir()

    for name in ["a.py", "b.py", "c.py"]:
        (d / name).write_text("# identical\ndef f(): pass\n")

    all_files, new_files, already = discover_files(d)
    assert len(all_files) == 3
    assert len(new_files) == 3

    ingest_many(new_files)

    # Simulate a fresh process by clearing the in-memory cache
    lss_store._file_cache.clear()

    all_files2, new_files2, already2 = discover_files(d)
    assert len(all_files2) == 3
    assert len(new_files2) == 0, (
        f"Expected 0 new files, got {len(new_files2)}: "
        f"{[str(f) for f in new_files2]}"
    )
    assert already2 == 3


def test_duplicate_content_unique_fts_entries(tmp_path):
    """Each duplicate-content file gets its own FTS chunk(s)."""
    d = tmp_path / "dup_fts"
    d.mkdir()
    for name in ["x.py", "y.py"]:
        (d / name).write_text("# shared code\ndef greet(): return 'hi'\n")

    ingest_many(d)

    con = sqlite3.connect(get_db_path())
    fts_uids = [
        r[0] for r in con.execute("SELECT DISTINCT file_uid FROM fts").fetchall()
    ]
    con.close()

    # Must be 2 distinct file_uids in FTS (one per file)
    assert len(fts_uids) == 2


# ── Mtime update on re-index (stale-mtime bug fix) ─────────────────────────


def test_mtime_updated_on_reindex(tmp_path):
    """When a file's mtime changes but content stays the same, the DB mtime
    must be updated so that discover_files still recognises it."""
    d = tmp_path / "mtime_proj"
    d.mkdir()
    f = d / "mod.py"
    f.write_text("def hello(): return 1\n")

    ensure_indexed(f)

    # Record the stored mtime
    con = sqlite3.connect(get_db_path())
    uid = con.execute("SELECT file_uid FROM files").fetchone()[0]
    mtime_before = con.execute(
        "SELECT mtime FROM files WHERE file_uid = ?", (uid,)
    ).fetchone()[0]
    con.close()

    # Touch the file forward (content unchanged, mtime changes)
    future = time.time() + 100
    os.utime(f, (future, future))
    lss_store._file_cache.clear()

    # Re-index (force because mtime changed => cache miss => full _do_index)
    ensure_indexed(f, force_reindex=True)

    con = sqlite3.connect(get_db_path())
    mtime_after = con.execute(
        "SELECT mtime FROM files WHERE file_uid = ?", (uid,)
    ).fetchone()[0]
    con.close()

    assert mtime_after != mtime_before, "DB mtime was not updated"
    assert abs(mtime_after - future) < 1.0, "DB mtime doesn't match new file mtime"


def test_discover_finds_file_after_mtime_touch(tmp_path):
    """After touching a file's mtime (content unchanged), discover_files must
    still recognise it as already-indexed on the next invocation."""
    d = tmp_path / "touch_proj"
    d.mkdir()
    f = d / "lib.py"
    f.write_text("x = 42\n")

    # Index it
    ingest_many(d)
    lss_store._file_cache.clear()

    # Verify it's recognised
    _, new1, already1 = discover_files(d)
    assert already1 == 1 and len(new1) == 0

    # Touch the file (simulates git checkout, cp, etc.)
    future = time.time() + 200
    os.utime(f, (future, future))
    lss_store._file_cache.clear()

    # File now has a new mtime — it will appear "new" to discover_files
    # because the DB still has the old mtime.
    _, new2, already2 = discover_files(d)

    if len(new2) == 1:
        # Re-index it (the _do_index fast-path should update the mtime)
        ingest_many(new2)
        lss_store._file_cache.clear()

        # NOW it must be recognised as already-indexed
        _, new3, already3 = discover_files(d)
        assert already3 == 1 and len(new3) == 0, (
            f"After re-index, expected 0 new but got {len(new3)}"
        )
    else:
        # If discover already found it (shouldn't happen with mtime change,
        # but accept it as correct)
        assert already2 == 1


# ── Migration: old content-based UIDs ────────────────────────────────────────


def test_migration_old_content_uid_still_discovered(tmp_path):
    """Files indexed with old content-based UIDs are still found by
    discover_files (which looks up by path+size+mtime+version, not uid)."""
    import hashlib as _hashlib
    import lss_config

    d = tmp_path / "migrate"
    d.mkdir()
    f = d / "old.py"
    f.write_text("def legacy(): return True\n")
    resolved = f.resolve()
    stat = resolved.stat()

    # Manually insert an old-style content-based UID
    content_sig = _hashlib.md5(resolved.read_bytes()).hexdigest()
    old_uid = f"f_{content_sig}"

    con = _init_db()
    con.execute(
        """INSERT INTO files
           (file_uid, path, size, mtime, content_sig, version, indexed_at, status)
           VALUES (?,?,?,?,?,?,?,?)""",
        (old_uid, str(resolved), stat.st_size, stat.st_mtime,
         content_sig, lss_config.VERSION_KEY, time.time(), "active"),
    )
    con.commit()
    con.close()

    lss_store._file_cache.clear()

    _, new_files, already = discover_files(d)
    assert already == 1
    assert len(new_files) == 0, "Old content-based entry should still be recognised"


def test_migration_reindex_replaces_old_uid(tmp_path):
    """When a file with an old content-based UID is re-indexed, the old entry
    is cleaned up and replaced with the new path-based UID."""
    import hashlib as _hashlib
    import lss_config

    d = tmp_path / "migrate_replace"
    d.mkdir()
    f = d / "module.py"
    f.write_text("def original(): pass\n")
    resolved = f.resolve()
    stat = resolved.stat()

    content_sig = _hashlib.md5(resolved.read_bytes()).hexdigest()
    old_uid = f"f_{content_sig}"
    new_uid = _path_uid(resolved)

    # They must differ (content-hash vs path-hash)
    assert old_uid != new_uid

    # Insert old-style entry
    con = _init_db()
    con.execute(
        """INSERT INTO files
           (file_uid, path, size, mtime, content_sig, version, indexed_at, status)
           VALUES (?,?,?,?,?,?,?,?)""",
        (old_uid, str(resolved), stat.st_size, stat.st_mtime,
         content_sig, lss_config.VERSION_KEY, time.time(), "active"),
    )
    con.commit()
    con.close()

    # Change file content so force_reindex actually re-indexes
    f.write_text("def updated(): return 'new'\n")
    future = time.time() + 50
    os.utime(f, (future, future))
    lss_store._file_cache.clear()

    returned_uid = ensure_indexed(f, force_reindex=True)
    assert returned_uid == new_uid

    # Old entry must be gone
    con = sqlite3.connect(get_db_path())
    old_row = con.execute(
        "SELECT 1 FROM files WHERE file_uid = ?", (old_uid,)
    ).fetchone()
    new_row = con.execute(
        "SELECT 1 FROM files WHERE file_uid = ?", (new_uid,)
    ).fetchone()
    con.close()

    assert old_row is None, "Old content-based UID was not cleaned up"
    assert new_row is not None, "New path-based UID was not inserted"
