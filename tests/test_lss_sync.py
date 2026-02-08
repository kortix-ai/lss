"""Tests for lss_sync — the file-watcher / debounced indexer."""

import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lss_sync import DebouncedIndexer, LSSSyncHandler, IGNORE_EXTENSIONS, IGNORE_DIRS

# We test the handler and indexer logic directly to avoid flaky timing issues
# from the full watchdog Observer pipeline on macOS.


# ── Helper: simulated FileSystemEvent ────────────────────────────────────────


class _FakeEvent:
    """Minimal stand-in for watchdog FileSystemEvent."""

    def __init__(self, src_path: str, is_directory: bool = False):
        self.src_path = src_path
        self.is_directory = is_directory
        self.event_type = "created"


class _FakeMovedEvent(_FakeEvent):
    def __init__(self, src_path: str, dest_path: str, is_directory: bool = False):
        super().__init__(src_path, is_directory)
        self.dest_path = dest_path
        self.event_type = "moved"


# ── LSSSyncHandler._should_ignore ─────────────────────────────────────────────


def test_handler_ignores_binary_extensions():
    """Extensions in IGNORE_EXTENSIONS must be ignored."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=10)
    handler = LSSSyncHandler(indexer)

    for ext in (".png", ".jpg", ".mp4", ".zip", ".pyc", ".db"):
        assert handler._should_ignore(f"/some/path/file{ext}") is True, ext


def test_handler_ignores_excluded_dirs():
    """Paths under IGNORE_DIRS must be ignored."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=10)
    handler = LSSSyncHandler(indexer)

    for d in ("__pycache__", ".git", "node_modules", ".venv"):
        assert handler._should_ignore(f"/project/{d}/foo.py") is True, d


def test_handler_ignores_hidden_files():
    """Hidden files (dotfiles) must be ignored."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=10)
    handler = LSSSyncHandler(indexer)

    assert handler._should_ignore("/project/.hidden_config") is True
    assert handler._should_ignore("/project/.env") is True


def test_handler_allows_normal_text_file():
    """Normal text files should not be ignored."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=10)
    handler = LSSSyncHandler(indexer)

    assert handler._should_ignore("/project/readme.md") is False
    assert handler._should_ignore("/project/src/main.py") is False


# ── DebouncedIndexer event collection ────────────────────────────────────────


def test_file_changed_adds_to_dirty():
    """file_changed() should add the path to _dirty_files."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)  # high debounce — won't fire
    indexer.file_changed("/project/foo.py")

    assert "/project/foo.py" in indexer._dirty_files


def test_file_deleted_adds_to_deleted():
    """file_deleted() should add the path to _deleted_files."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    indexer.file_deleted("/project/bar.py")

    assert "/project/bar.py" in indexer._deleted_files


def test_file_changed_removes_from_deleted():
    """If a file reappears after deletion, it should move from deleted to dirty."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    indexer.file_deleted("/project/foo.py")
    indexer.file_changed("/project/foo.py")

    assert "/project/foo.py" in indexer._dirty_files
    assert "/project/foo.py" not in indexer._deleted_files


def test_file_deleted_removes_from_dirty():
    """If a dirty file is deleted, it should move from dirty to deleted."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    indexer.file_changed("/project/foo.py")
    indexer.file_deleted("/project/foo.py")

    assert "/project/foo.py" not in indexer._dirty_files
    assert "/project/foo.py" in indexer._deleted_files


# ── LSSSyncHandler event routing ──────────────────────────────────────────────


def test_on_created_routes_to_dirty():
    """on_created should feed the path to the indexer."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    handler = LSSSyncHandler(indexer)

    handler.on_created(_FakeEvent("/project/new_file.txt"))
    assert "/project/new_file.txt" in indexer._dirty_files


def test_on_created_ignores_directories():
    """on_created should skip directory events."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    handler = LSSSyncHandler(indexer)

    handler.on_created(_FakeEvent("/project/new_dir", is_directory=True))
    assert len(indexer._dirty_files) == 0


def test_on_deleted_routes_to_deleted():
    """on_deleted should feed the path to the indexer's deleted set."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    handler = LSSSyncHandler(indexer)

    handler.on_deleted(_FakeEvent("/project/old_file.txt"))
    assert "/project/old_file.txt" in indexer._deleted_files


def test_on_created_ignores_binary():
    """on_created should ignore files with binary extensions."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    handler = LSSSyncHandler(indexer)

    handler.on_created(_FakeEvent("/project/image.png"))
    assert len(indexer._dirty_files) == 0


def test_on_created_ignores_hidden():
    """on_created should ignore hidden files."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    handler = LSSSyncHandler(indexer)

    handler.on_created(_FakeEvent("/project/.secret"))
    assert len(indexer._dirty_files) == 0


def test_on_modified_routes_to_dirty():
    """on_modified should add the file to dirty."""
    indexer = DebouncedIndexer(["/tmp/watch"], debounce=999)
    handler = LSSSyncHandler(indexer)

    handler.on_modified(_FakeEvent("/project/updated.py"))
    assert "/project/updated.py" in indexer._dirty_files


# ── Debounce batching (unit-level) ───────────────────────────────────────────


def test_debounce_batches_rapid_changes():
    """Multiple rapid file_changed() calls should result in one _do_index call."""
    call_count = {"n": 0}
    original_do_index = DebouncedIndexer._do_index

    def counting_do_index(self, dirty, deleted):
        call_count["n"] += 1

    indexer = DebouncedIndexer(["/tmp/watch"], debounce=0.3)
    indexer._do_index = counting_do_index.__get__(indexer)

    # Simulate 5 rapid file changes
    for i in range(5):
        indexer.file_changed(f"/project/file{i}.txt")
        time.sleep(0.05)

    # Wait for debounce + margin
    time.sleep(0.8)

    # Should have batched into a single index pass
    assert call_count["n"] == 1
    # All 5 files should have been in the dirty set at trigger time
    # (they were cleared by _trigger_index)
    assert len(indexer._dirty_files) == 0  # cleared after trigger
