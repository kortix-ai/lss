"""Tests for lss_cli — the command-line interface."""

import json
import os
import sys
from io import StringIO

import pytest

from lss_cli import main, __version__, _C
from lss_store import ensure_indexed, get_db_path, _init_db


# Disable colors in tests for predictable output matching
@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)  # reset to auto-detect


# ── version ──────────────────────────────────────────────────────────────────


def test_cli_version(capsys):
    """'lss version' should print the version string and return 0."""
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert __version__ in out


# ── index ────────────────────────────────────────────────────────────────────


def test_cli_index_file(sample_dir, capsys):
    """'lss index <file>' should succeed for a valid text file."""
    rc = main(["index", str(sample_dir / "readme.md")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "indexed" in out.lower() or "Indexed" in out


def test_cli_index_nonexistent(capsys):
    """'lss index' on a non-existent path should return exit code 2."""
    rc = main(["index", "/no/such/file.txt"])
    assert rc == 2


# ── ls ───────────────────────────────────────────────────────────────────────


def test_cli_ls_empty(capsys):
    """'lss ls' with no indexed files should show empty state."""
    con = _init_db()
    con.close()
    rc = main(["ls"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no files" in out.lower() or "(empty)" in out.lower() or "No files" in out


def test_cli_ls_after_index(sample_dir, capsys):
    """After indexing a file, 'lss ls' should list it."""
    ensure_indexed(sample_dir / "readme.md")
    rc = main(["ls"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "readme.md" in out


# ── sweep ────────────────────────────────────────────────────────────────────


def test_cli_sweep_clear_all(sample_dir, capsys):
    """'lss sweep --clear-all' should succeed and clear the database."""
    from lss_store import ingest_many

    ingest_many(sample_dir)
    rc = main(["sweep", "--clear-all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "clear" in out.lower() or "Clear" in out


# ── db-path ──────────────────────────────────────────────────────────────────


def test_cli_dbpath(capsys):
    """'lss db-path' should print a path containing 'lss.db'."""
    rc = main(["db-path"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lss.db" in out


# ── search (requires OpenAI) ────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_cli_search(sample_dir, capsys):
    """'lss search <query> -p <dir>' should return 0."""
    rc = main(["search", "deployment", "-p", str(sample_dir)])
    assert rc == 0


# ── smart routing (no subcommand = search) ───────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_cli_implicit_search(sample_dir, capsys):
    """'lss <query> <path>' should work without 'search' subcommand."""
    rc = main(["deployment", str(sample_dir)])
    assert rc == 0
