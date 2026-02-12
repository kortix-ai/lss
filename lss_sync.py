#!/usr/bin/env python3
"""
lss-sync — File-watcher daemon that keeps the semantic search index always
up-to-date by reacting to filesystem changes in real time.

Uses watchdog (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on
Windows) to detect file creates/modifies/deletes and triggers re-indexing with
a debounce window so rapid bursts of changes are batched into a single index
pass.

Can be installed as a standalone CLI:
    pip install .          # from the lss/ directory
    lss-sync --watch /path/to/dir1 --watch /path/to/dir2

Or run directly:
    python lss_sync.py --watch ~/Desktop --watch ~/.kortix

Environment variables:
    LSS_DIR             — lss database directory (default: ~/.lss)
    OPENAI_API_KEY      — Required for embedding generation
"""

import argparse
import fnmatch
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Ensure LSS_DIR is set before importing lss_store (reads config at import time)
if "LSS_DIR" not in os.environ:
    os.environ["LSS_DIR"] = os.path.expanduser("~/.lss")

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

# ── Configuration ───────────────────────────────────────────────────────────

# Debounce window in seconds — after a file change, wait this long for more
# changes before triggering re-index. Prevents hammering the embeddings API
# when many files change at once (e.g. git checkout, bulk save).
DEFAULT_DEBOUNCE = 2.0

# Minimum seconds between full index passes (safety valve)
MIN_INDEX_INTERVAL = 10.0

# Periodic sweep interval (seconds) — clean up stale entries
SWEEP_INTERVAL = 3600  # 1 hour

# File extensions to ignore (not useful for semantic search)
IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".class", ".o", ".so", ".dylib", ".dll",
    ".db", ".db-shm", ".db-wal", ".sqlite", ".sqlite3",
    ".lock", ".log",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".whl", ".egg",
}

# Directory names to ignore
IGNORE_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".eggs", ".cache",
}

BROWSER_CACHE_DIR_MARKERS = {
    "cache", "cache_data", "code cache", "gpucache",
    "shadercache", "grshadercache", "graphitedawncache",
}

BROWSER_DIR_MARKERS = {
    "chromium", "google-chrome", "chrome", "brave", "edge", "msedge", "electron",
}


def _is_browser_cache_path(path: Path) -> bool:
    """Return True if path looks like browser cache churn."""
    parts_lower = {part.lower() for part in path.parts}
    if not (parts_lower & BROWSER_DIR_MARKERS):
        return False
    return bool(parts_lower & BROWSER_CACHE_DIR_MARKERS)


def _matches_exclude_pattern(path: Path, pattern: str) -> bool:
    """Evaluate one exclude pattern against a path."""
    if "*" in pattern:
        return fnmatch.fnmatch(path.name, pattern)
    if "/" in pattern:
        return pattern in str(path)
    return pattern in path.parts


def _should_ignore_path(
    path: str | bytes,
    watch_paths: list[str],
    exclude_patterns: list[str] | None = None,
    max_depth: int | None = None,
) -> bool:
    """Shared path-ignore logic for handler + indexer."""
    p = Path(os.fsdecode(path))

    if p.suffix.lower() in IGNORE_EXTENSIONS:
        return True

    for part in p.parts:
        if part in IGNORE_DIRS:
            return True

    if p.name.startswith(".") and p.name not in {".kortix"}:
        return True

    if "lss.db" in p.name or ".lss" in str(p):
        return True

    if _is_browser_cache_path(p):
        return True

    try:
        from lss_store import _should_exclude_path
        for watch_path in watch_paths:
            if _should_exclude_path(str(p), watch_path):
                return True
    except Exception:
        pass

    for pattern in (exclude_patterns or []):
        if _matches_exclude_pattern(p, pattern):
            return True

    if max_depth is not None:
        for watch_path in watch_paths:
            try:
                rel = p.relative_to(watch_path)
                if len(rel.parts) > max_depth:
                    return True
            except ValueError:
                continue

    return False

# Graceful shutdown
_running = True


def handle_signal(signum, _frame):
    global _running
    _running = False
    print(f"[lss-sync] Received signal {signum}, shutting down...", flush=True)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ── Debounced Indexer ───────────────────────────────────────────────────────

class DebouncedIndexer:
    """Collects file-change events and triggers indexing after a debounce window.
    
    When a file change is detected, the indexer waits `debounce` seconds for
    more changes. If no new changes arrive within the window, it triggers
    re-indexing of all changed paths. If more changes arrive, the timer resets.
    
    This prevents re-indexing on every keystroke while ensuring changes are
    picked up within seconds.
    """

    def __init__(self, watch_paths: list[str], debounce: float = DEFAULT_DEBOUNCE,
                 exclude_patterns: list[str] | None = None, max_depth: int | None = None):
        self.watch_paths = watch_paths
        self.debounce = debounce
        self.exclude_patterns = exclude_patterns or []
        self.max_depth = max_depth
        self._dirty_files: set[str] = set()
        self._deleted_files: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._last_index_time = 0.0
        self._indexing = False

    def file_changed(self, path: str | bytes):
        """Called when a file is created or modified."""
        normalized = os.fsdecode(path)
        with self._lock:
            self._dirty_files.add(normalized)
            self._deleted_files.discard(normalized)
            self._schedule_index()

    def file_deleted(self, path: str | bytes):
        """Called when a file is deleted."""
        normalized = os.fsdecode(path)
        with self._lock:
            self._deleted_files.add(normalized)
            self._dirty_files.discard(normalized)
            self._schedule_index()

    def _schedule_index(self):
        """Schedule an index pass after the debounce window."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce, self._trigger_index)
        self._timer.daemon = True
        self._timer.start()

    def _trigger_index(self):
        """Actually run the index pass."""
        if self._indexing:
            # Already indexing — reschedule
            self._schedule_index()
            return

        # Enforce minimum interval
        now = time.time()
        if now - self._last_index_time < MIN_INDEX_INTERVAL:
            remaining = MIN_INDEX_INTERVAL - (now - self._last_index_time)
            self._timer = threading.Timer(remaining, self._trigger_index)
            self._timer.daemon = True
            self._timer.start()
            return

        # Grab the pending changes
        with self._lock:
            dirty = self._dirty_files.copy()
            deleted = self._deleted_files.copy()
            self._dirty_files.clear()
            self._deleted_files.clear()
            self._timer = None

        if not dirty and not deleted:
            return

        self._indexing = True
        self._last_index_time = time.time()

        try:
            self._do_index(dirty, deleted)
        except Exception as e:
            print(f"[lss-sync] Index error: {e}", flush=True)
        finally:
            self._indexing = False

    def _do_index(self, dirty: set[str], deleted: set[str]):
        """Perform the actual indexing work."""
        try:
            from lss_store import ensure_indexed, remove_files, _is_text_file
        except ImportError as e:
            print(f"[lss-sync] Failed to import lss_store: {e}", flush=True)
            return

        t0 = time.time()
        indexed = 0
        removed = 0
        skipped = 0
        errors = 0

        # Index changed files
        for fpath in dirty:
            p = Path(fpath)
            if _should_ignore_path(
                str(p), self.watch_paths, self.exclude_patterns, self.max_depth
            ):
                skipped += 1
                continue
            if not p.exists() or not p.is_file():
                continue
            if not _is_text_file(p):
                skipped += 1
                continue
            try:
                ensure_indexed(p)
                indexed += 1
            except ValueError as e:
                if "Not a text file" in str(e):
                    skipped += 1
                    continue
                print(f"[lss-sync] Index {fpath} failed: {e}", flush=True)
                errors += 1
            except Exception as e:
                print(f"[lss-sync] Index {fpath} failed: {e}", flush=True)
                errors += 1

        # Handle deleted files
        if deleted:
            try:
                remove_candidates = [
                    str(p)
                    for p in deleted
                    if not _should_ignore_path(
                        str(p), self.watch_paths, self.exclude_patterns, self.max_depth
                    )
                ]
                if remove_candidates:
                    remove_files(remove_candidates)
                    removed = len(remove_candidates)
            except Exception as e:
                print(f"[lss-sync] Remove failed: {e}", flush=True)
                errors += 1

        elapsed = time.time() - t0
        print(
            f"[lss-sync] Indexed {indexed}, removed {removed}, skipped {skipped} "
            f"({errors} errors) in {elapsed:.1f}s",
            flush=True,
        )

    def force_full_index(self):
        """Force a full index of all watched paths (used on startup)."""
        try:
            from lss_store import ingest_many, ensure_indexed
        except ImportError as e:
            print(f"[lss-sync] Failed to import lss_store: {e}", flush=True)
            return

        t0 = time.time()
        for path_str in self.watch_paths:
            p = Path(path_str)
            if not p.exists():
                os.makedirs(path_str, exist_ok=True)
                continue
            try:
                if p.is_dir():
                    ingest_many(p)
                    print(f"[lss-sync] Indexed directory {path_str}", flush=True)
                elif p.is_file():
                    ensure_indexed(p)
                    print(f"[lss-sync] Indexed file {path_str}", flush=True)
            except Exception as e:
                print(f"[lss-sync] Index {path_str} error: {e}", flush=True)

        elapsed = time.time() - t0
        self._last_index_time = time.time()
        print(f"[lss-sync] Full index complete in {elapsed:.1f}s", flush=True)


# ── File System Event Handler ───────────────────────────────────────────────

class LSSSyncHandler(FileSystemEventHandler):
    """Watchdog event handler that feeds file changes to the DebouncedIndexer."""

    def __init__(self, indexer: DebouncedIndexer, exclude_patterns: list[str] | None = None,
                 max_depth: int | None = None):
        super().__init__()
        self.indexer = indexer
        self.exclude_patterns = exclude_patterns or []
        self.max_depth = max_depth

    def _should_ignore(self, path: str | bytes) -> bool:
        """Check if this path should be ignored."""
        return _should_ignore_path(
            path,
            self.indexer.watch_paths,
            self.exclude_patterns,
            self.max_depth,
        )

    def on_created(self, event: FileSystemEvent):
        if event.is_directory or self._should_ignore(event.src_path):
            return
        self.indexer.file_changed(event.src_path)

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory or self._should_ignore(event.src_path):
            return
        self.indexer.file_changed(event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory or self._should_ignore(event.src_path):
            return
        self.indexer.file_deleted(event.src_path)

    def on_moved(self, event: FileSystemEvent):
        if event.is_directory:
            return
        # Treat as delete old + create new
        if not self._should_ignore(event.src_path):
            self.indexer.file_deleted(event.src_path)
        if hasattr(event, "dest_path") and not self._should_ignore(event.dest_path):
            self.indexer.file_changed(event.dest_path)


# ── Sweep Thread ────────────────────────────────────────────────────────────

def sweep_loop():
    """Periodically sweep stale entries from the database."""
    while _running:
        # Sleep in small increments for responsive shutdown
        for _ in range(SWEEP_INTERVAL):
            if not _running:
                return
            time.sleep(1)

        if not _running:
            return

        try:
            from lss_store import sweep as lss_sweep
            lss_sweep(retention_days=180)
            print("[lss-sync] Sweep complete", flush=True)
        except Exception as e:
            print(f"[lss-sync] Sweep error: {e}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="lss-sync",
        description="File-watcher daemon that keeps the semantic search index in sync.",
    )
    parser.add_argument(
        "--watch", "-w",
        action="append",
        default=[],
        help="Path to watch for changes (can be specified multiple times)",
    )
    parser.add_argument(
        "--exclude", "-e",
        action="append",
        default=[],
        help="Exclusion pattern (glob for *, path substring for /, dir component otherwise)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum recursion depth for watching directories",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=DEFAULT_DEBOUNCE,
        help=f"Debounce window in seconds (default: {DEFAULT_DEBOUNCE})",
    )
    parser.add_argument(
        "--no-initial-index",
        action="store_true",
        help="Skip full index on startup",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=0,
        help="Seconds to wait before starting (useful for s6/systemd ordering)",
    )
    args = parser.parse_args()

    watch_paths = args.watch
    exclude_patterns = args.exclude

    # If no --watch paths given, read from config.json
    if not watch_paths:
        try:
            import lss_config
            cfg = lss_config.load_config()
            watch_paths = cfg.get("watch_paths", [])
            # Merge config exclusions with CLI exclusions
            cfg_excludes = cfg.get("exclude_patterns", [])
            exclude_patterns = list(set(exclude_patterns + cfg_excludes))
        except Exception:
            pass

    if not watch_paths:
        print("[lss-sync] No watch paths specified.", file=sys.stderr)
        print("[lss-sync] Use --watch <path> or configure with: lss watch add <path>", file=sys.stderr)
        sys.exit(1)

    # Resolve paths
    watch_paths = [str(Path(p).resolve()) for p in watch_paths]

    # Ensure all watch paths exist
    for p in watch_paths:
        os.makedirs(p, exist_ok=True)

    print(f"[lss-sync] Starting file-watcher daemon", flush=True)
    print(f"[lss-sync] Watch paths: {watch_paths}", flush=True)
    print(f"[lss-sync] Debounce: {args.debounce}s", flush=True)
    print(f"[lss-sync] LSS_DIR: {os.environ.get('LSS_DIR', '(default)')}", flush=True)
    if exclude_patterns:
        print(f"[lss-sync] Exclude patterns: {exclude_patterns}", flush=True)
    if args.max_depth is not None:
        print(f"[lss-sync] Max depth: {args.max_depth}", flush=True)

    if args.startup_delay > 0:
        print(f"[lss-sync] Waiting {args.startup_delay}s for other services...", flush=True)
        time.sleep(args.startup_delay)

    # Create the debounced indexer
    indexer = DebouncedIndexer(
        watch_paths, debounce=args.debounce,
        exclude_patterns=exclude_patterns, max_depth=args.max_depth,
    )

    # Start the watchdog observer FIRST — so file changes are captured
    # immediately, even while the initial index is still running.
    observer = Observer()
    fs_handler = LSSSyncHandler(
        indexer, exclude_patterns=exclude_patterns, max_depth=args.max_depth,
    )

    for path in watch_paths:
        observer.schedule(fs_handler, path, recursive=True)
        print(f"[lss-sync] Watching: {path}", flush=True)

    observer.start()

    # Start sweep thread
    sweep_thread = threading.Thread(target=sweep_loop, daemon=True)
    sweep_thread.start()

    print("[lss-sync] Daemon ready. Watching for changes...", flush=True)

    # Run initial full index in a background thread so we don't block
    # the file watcher. This can take minutes for large codebases
    # (embedding generation via OpenAI API).
    if not args.no_initial_index:
        def _initial_index():
            try:
                print("[lss-sync] Running initial full index (background)...", flush=True)
                indexer.force_full_index()
                print("[lss-sync] Initial index complete.", flush=True)
            except Exception as e:
                print(f"[lss-sync] Initial index error: {e}", flush=True)

        init_thread = threading.Thread(target=_initial_index, daemon=True)
        init_thread.start()

    # Main loop — just keep alive and respond to signals
    try:
        while _running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("[lss-sync] Shutting down...", flush=True)
        observer.stop()
        observer.join(timeout=5)
        print("[lss-sync] Daemon stopped.", flush=True)


if __name__ == "__main__":
    main()
