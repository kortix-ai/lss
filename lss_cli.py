#!/usr/bin/env python3
"""
lss — Local Semantic Search

Usage:
    lss <query>                    search current directory
    lss <query> -p <path>          search a specific file or directory
    lss <query> <path>             also works — path auto-detected
    lss "auth JWT" "deploy k8s"    multiple queries
    lss <query> --json             machine-readable output
    lss <query> | head             colors auto-off when piped

Management:
    lss status                     show config, watched paths, DB stats
    lss index <path>               manually index a file
    lss ls                         list indexed files
    lss watch add|remove|list      manage watched directories
    lss exclude add|remove|list    manage exclusion patterns
    lss sweep                      cleanup / maintenance
    lss eval                       run search quality evaluation
    lss eval --json                evaluation results as JSON
"""

import argparse, sys, json, os, sqlite3, time
from pathlib import Path
from typing import List

__version__ = "0.4.0"

# Set debug BEFORE importing other modules
import lss_config

# flat imports
from lss_store import (
    ensure_indexed,
    get_db_path,
    remove_files,
    sweep as sweep_default,
    clear_all,
    clear_embeddings,
)


# ── Color & formatting ──────────────────────────────────────────────────────

class _C:
    """ANSI colors with auto-detection. Colors ON for TTY, OFF when piped."""
    _enabled = None

    @classmethod
    def enabled(cls):
        if cls._enabled is None:
            cls._enabled = (
                hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )
        return cls._enabled

    @classmethod
    def set_enabled(cls, val):
        cls._enabled = val

    @classmethod
    def _wrap(cls, code, text):
        if cls.enabled():
            return f"\033[{code}m{text}\033[0m"
        return str(text)

    @classmethod
    def bold(cls, t):    return cls._wrap("1", t)
    @classmethod
    def dim(cls, t):     return cls._wrap("2", t)
    @classmethod
    def green(cls, t):   return cls._wrap("32", t)
    @classmethod
    def yellow(cls, t):  return cls._wrap("33", t)
    @classmethod
    def blue(cls, t):    return cls._wrap("34", t)
    @classmethod
    def magenta(cls, t): return cls._wrap("35", t)
    @classmethod
    def cyan(cls, t):    return cls._wrap("36", t)
    @classmethod
    def red(cls, t):     return cls._wrap("31", t)
    @classmethod
    def bold_green(cls, t):  return cls._wrap("1;32", t)
    @classmethod
    def bold_magenta(cls, t): return cls._wrap("1;35", t)
    @classmethod
    def bold_cyan(cls, t): return cls._wrap("1;36", t)
    @classmethod
    def bold_yellow(cls, t): return cls._wrap("1;33", t)
    @classmethod
    def bold_red(cls, t): return cls._wrap("1;31", t)


def _human_size(nbytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _relative_path(abs_path, base=None):
    """Display path relative to base (or cwd) when possible."""
    try:
        if base is None:
            base = Path.cwd()
        else:
            base = Path(base).resolve()
        rel = Path(abs_path).resolve().relative_to(base)
        return str(rel)
    except (ValueError, RuntimeError):
        return str(abs_path)


def _time_ago(ts):
    """Human-friendly relative time string."""
    ago = time.time() - ts
    if ago < 60:
        return f"{ago:.0f}s ago"
    elif ago < 3600:
        return f"{ago / 60:.0f}m ago"
    elif ago < 86400:
        return f"{ago / 3600:.1f}h ago"
    else:
        return f"{ago / 86400:.1f}d ago"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_queries(args) -> List[str]:
    qs = []
    if getattr(args, "queries", None):
        qs.extend(args.queries)

    qfile = getattr(args, "qfile", None)
    if qfile:
        if qfile == "-":
            for line in sys.stdin:
                line = line.strip()
                if line and not line.startswith("#"):
                    qs.append(line)
        else:
            with open(qfile, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        qs.append(line)

    return list(dict.fromkeys(qs))  # de-dupe, preserve order


def _check_openai_key():
    """Check if OPENAI_API_KEY is set, print helpful error if not."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print(_C.bold_red("error:") + " OPENAI_API_KEY is not set\n", file=sys.stderr)
        print("LSS uses OpenAI embeddings for semantic search.", file=sys.stderr)
        print("Set your API key:", file=sys.stderr)
        print(f"  {_C.bold('export OPENAI_API_KEY=sk-...')}", file=sys.stderr)
        print(f"\nOr add it to {_C.dim('~/.lss/.env')} or {_C.dim('~/.config/lss/.env')}", file=sys.stderr)
        print(f"Get a key at: {_C.cyan('https://platform.openai.com/api-keys')}", file=sys.stderr)
        return False
    return True


def _is_interactive():
    """True when stdout is a TTY (not piped, not redirected)."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _confirm_index(new_count, total_count, already_count, path_label):
    """Ask the user to confirm indexing.  Returns True to proceed."""
    print(
        _C.dim("found")
        + f" {_C.bold(str(total_count))} text file{'s' if total_count != 1 else ''}"
        + f" in {_C.bold(path_label)}"
        + (f"  {_C.dim('(' + str(already_count) + ' already indexed, ' + str(new_count) + ' new)')}"
           if already_count else ""),
        file=sys.stderr,
    )
    try:
        answer = input(f"  Index {new_count} new file{'s' if new_count != 1 else ''}? [Y/n] ")
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False
    return answer.strip().lower() in ("", "y", "yes")


def _progress_line(current, total, path, base_path=None):
    """Overwrite the current terminal line with indexing progress."""
    rel = _relative_path(path, base_path)
    # Truncate filename to fit terminal width
    try:
        cols = os.get_terminal_size().columns
    except (AttributeError, ValueError, OSError):
        cols = 80
    prefix = f"  indexing [{current}/{total}] "
    max_name = cols - len(prefix) - 1
    if len(rel) > max_name and max_name > 3:
        rel = "..." + rel[-(max_name - 3):]
    line = f"{prefix}{rel}"
    sys.stderr.write(f"\r{line:<{cols}}")
    sys.stderr.flush()
    if current == total:
        sys.stderr.write("\r" + " " * cols + "\r")  # clear line when done
        sys.stderr.flush()


def _resolve_search_args(args):
    """Smart argument resolution for search.

    Handles these cases:
        lss "query"                     -> path=., queries=["query"]
        lss "query" -p ~/docs           -> path=~/docs, queries=["query"]
        lss "query" ~/docs              -> path=~/docs, queries=["query"]
        lss "q1" "q2" ~/docs            -> path=~/docs, queries=["q1", "q2"]
        lss "q1" "q2"                   -> path=., queries=["q1", "q2"]
        lss search "query"              -> path=., queries=["query"]  (compat)
        lss search ~/docs "query"       -> path=~/docs, queries=["query"]  (compat)
    """
    queries = list(args.queries) if args.queries else []
    path = getattr(args, "path", None)

    # If -p/--path was given explicitly, use it
    if path:
        return Path(path).resolve(), queries

    # Smart detection: if there are 2+ positional args and the last one
    # looks like an existing path, treat it as the search path.
    # With only 1 arg, always treat it as a query (use -p for path).
    if len(queries) >= 2:
        last = queries[-1]
        candidate = Path(last).expanduser()
        if candidate.exists() and (candidate.is_dir() or candidate.is_file()):
            # It's a path — remove from queries, use as path
            queries = queries[:-1]
            return candidate.resolve(), queries

    # Default to current directory
    return Path.cwd().resolve(), queries


# ── Search command ───────────────────────────────────────────────────────────

def cmd_search(args) -> int:
    from lss_store import ingest_many, get_file_uid, _is_text_file
    from semantic_search import semantic_search

    # Check for API key early
    if not _check_openai_key():
        return 1

    # Resolve path and queries
    path, queries = _resolve_search_args(args)

    # Also read from -Q/--qfile
    qfile = getattr(args, "qfile", None)
    if qfile:
        if qfile == "-":
            for line in sys.stdin:
                line = line.strip()
                if line and not line.startswith("#"):
                    queries.append(line)
        else:
            with open(qfile, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        queries.append(line)

    # De-dupe
    queries = list(dict.fromkeys(queries))

    if not queries:
        print(_C.bold_red("error:") + " no query provided\n", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print(f'  {_C.bold("lss")} {_C.green(chr(34) + "your search query" + chr(34))}', file=sys.stderr)
        print(f'  {_C.bold("lss")} {_C.green(chr(34) + "query" + chr(34))} {_C.cyan("-p ~/Documents")}', file=sys.stderr)
        print(f'  {_C.bold("lss")} {_C.green(chr(34) + "query" + chr(34))} {_C.cyan("~/Documents")}', file=sys.stderr)
        print(f'  {_C.bold("lss")} {_C.green(chr(34) + "q1" + chr(34) + " " + chr(34) + "q2" + chr(34))} {_C.dim("# multiple queries")}', file=sys.stderr)
        return 2

    if not path.exists():
        print(
            _C.bold_red("error:") + f" path not found: {path}\n",
            file=sys.stderr,
        )
        print(f"Make sure the path exists, or omit it to search the current directory.", file=sys.stderr)
        return 2

    try:
        from lss_store import discover_files

        no_index = getattr(args, "no_index", False) or getattr(args, "indexed_only", False)
        auto_yes = getattr(args, "yes", False)
        use_json = getattr(args, "json", False)
        unindexed_files = []

        # Handle indexed-only / no-index mode
        if no_index:
            if path.is_file():
                file_uid = get_file_uid(path)
                if not file_uid:
                    unindexed_files.append(str(path))
            elif path.is_dir():
                all_files = []
                for file_path in path.rglob("*"):
                    if file_path.is_file() and _is_text_file(file_path):
                        all_files.append(file_path)
                for file_path in all_files:
                    if not get_file_uid(file_path):
                        unindexed_files.append(str(file_path))
        else:
            # Normal mode: auto-index with confirmation + progress
            if path.is_file():
                ensure_indexed(path)
            elif path.is_dir():
                all_files, new_files, already = discover_files(path)
                total = len(all_files)
                new_count = len(new_files)

                if new_count > 0:
                    scope_label = _relative_path(path) if path != Path.cwd().resolve() else "."
                    interactive = _is_interactive() and not use_json and not auto_yes

                    if interactive:
                        if not _confirm_index(new_count, total, already, scope_label):
                            print(_C.dim("  skipped indexing"), file=sys.stderr)
                            # Fall through to search whatever is already indexed
                            pass
                        else:
                            # User confirmed — index only the NEW files (not the whole dir)
                            def _prog(cur, tot, p):
                                _progress_line(cur, tot, p, path)
                            ingest_many(new_files, progress_cb=_prog)
                            print(
                                _C.green(f"  indexed {new_count} file{'s' if new_count != 1 else ''}"),
                                file=sys.stderr,
                            )
                    else:
                        # Non-interactive / --json / --yes: index only new files
                        if not use_json and _is_interactive():
                            def _prog(cur, tot, p):
                                _progress_line(cur, tot, p, path)
                            ingest_many(new_files, progress_cb=_prog)
                            print(
                                _C.green(f"  indexed {new_count} file{'s' if new_count != 1 else ''}"),
                                file=sys.stderr,
                            )
                        else:
                            ingest_many(new_files)
                elif total > 0:
                    # All files already indexed — proceed silently
                    pass
                # else: empty directory — proceed to search (will return 0 hits)
            else:
                print(_C.bold_red("error:") + f" not a file or directory: {path}", file=sys.stderr)
                return 2

        t0 = time.time()
        batches = semantic_search(str(path), queries, indexed_only=no_index)
        dt = time.time() - t0

        # ── JSON output ──────────────────────────────────────────────────
        if args.json:
            payload = []
            for q, hits in zip(queries, batches):
                clean_hits = []
                for h in (hits[: args.limit] if hits else []):
                    clean_hits.append({
                        "file_path": h.get("file_path"),
                        "score": round(h.get("score", 0.0), 4),
                        "snippet": h.get("snippet"),
                        "rank_stage": h.get("rank_stage"),
                        "indexed_at": h.get("indexed_at"),
                    })
                query_result = {"query": q, "hits": clean_hits}
                if unindexed_files:
                    query_result["unindexed_files"] = unindexed_files
                payload.append(query_result)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        # ── Human output (beautiful terminal display) ────────────────────

        # Header line
        scope = _relative_path(path) if path != Path.cwd().resolve() else "."
        is_dir = path.is_dir()
        header_parts = []
        if is_dir:
            header_parts.append(f"{_C.dim('searching')} {_C.bold(scope + '/')}")
        else:
            header_parts.append(f"{_C.dim('searching')} {_C.bold(scope)}")
        header_parts.append(_C.dim(f"{dt:.2f}s"))
        print(" ".join(header_parts))

        if unindexed_files and len(unindexed_files) > 0:
            print(_C.yellow(f"  {len(unindexed_files)} files not yet indexed") +
                  _C.dim(f" (run without --no-index to auto-index)"))

        total_hits = 0

        for qi, (q, hits) in enumerate(zip(queries, batches)):
            # Query header (only shown if multiple queries)
            if len(queries) > 1:
                print()
                print(_C.bold_cyan(f"query: ") + _C.bold(q))

            if not hits:
                print()
                print(_C.dim("  no results"))
                if is_dir:
                    print(_C.dim("  try a broader query or different directory"))
                continue

            limited_hits = hits[:args.limit]
            total_hits += len(limited_hits)

            # Group hits by file
            file_groups = {}
            for h in limited_hits:
                fp = h.get("file_path", "")
                if fp not in file_groups:
                    file_groups[fp] = []
                file_groups[fp].append(h)

            print()
            for fp, group in file_groups.items():
                # File header (relative path, bold + colored)
                rel = _relative_path(fp, path if is_dir else path.parent)
                print(_C.bold_magenta(rel))

                for h in group:
                    score = h.get("score", 0.0)
                    snippet = (h.get("snippet", "") or "").replace("\n", " ").strip()

                    # Score bar: visual relevance indicator
                    score_str = _C.dim(f"  {score:.3f}")

                    # Snippet with query term highlighting
                    if snippet:
                        print(f"{score_str}  {snippet}")
                    else:
                        print(f"{score_str}  {_C.dim('(no snippet)')}")

                print()  # blank line between files

        # Footer
        if total_hits == 0 and not any(hits for _, hits in zip(queries, batches)):
            pass  # already printed "no results" above
        elif len(queries) == 1:
            count_str = f"{total_hits} result{'s' if total_hits != 1 else ''}"
            print(_C.dim(f"{count_str}"))

        return 0

    except ValueError as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg:
            _check_openai_key()
            return 1
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        return 2


# ── Management commands ──────────────────────────────────────────────────────

def cmd_sweep(args) -> int:
    """Maintenance: clean up indexed data."""
    try:
        if args.clear_all:
            clear_all()
            print("Cleared database (schema recreated)")

        elif args.remove:
            n = remove_files(args.remove)
            if n == 0:
                print("Nothing removed (no matching paths)")
            else:
                print(f"Removed {n} file{'s' if n != 1 else ''}")

        elif args.clear_embeddings is not None:
            days = None if args.clear_embeddings == 0 else args.clear_embeddings
            clear_embeddings(days=days)
            label = "all" if days is None else f"older than {days}d"
            print(f"Cleared embeddings ({label})")

        else:
            sweep_default(retention_days=args.retention_days)
            print("Sweep complete")

        # optional housekeeping — best-effort, non-blocking
        if not args.no_optimize:
            try:
                con = sqlite3.connect(get_db_path(), check_same_thread=False, timeout=30)
                con.execute("PRAGMA busy_timeout=30000")
                try:
                    con.execute("INSERT INTO fts(fts) VALUES('optimize')")
                    con.commit()
                except sqlite3.OperationalError:
                    pass
                try:
                    con.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.OperationalError:
                    pass
                con.close()
            except Exception:
                pass

        return 0
    except Exception as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        return 2


def cmd_dbpath(_args) -> int:
    try:
        print(get_db_path())
        return 0
    except Exception as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        return 2


def cmd_ls(_args) -> int:
    """List what's indexed."""
    try:
        db = get_db_path()
        if not Path(db).exists():
            print(_C.dim("No files indexed yet."))
            print(f"Index a directory:  {_C.bold('lss')} {_C.green(chr(34) + 'query' + chr(34))} {_C.cyan('~/Documents')}")
            return 0

        con = sqlite3.connect(db, check_same_thread=False, timeout=30)
        con.execute("PRAGMA busy_timeout=30000")
        cur = con.cursor()
        rows = cur.execute(
            "SELECT file_uid, path, size, mtime, version, status "
            "FROM files ORDER BY status DESC, path"
        ).fetchall()
        con.close()
        if not rows:
            print(_C.dim("No files indexed yet."))
            return 0

        for uid, fpath, size, mtime, version, status in rows:
            ts = time.strftime("%Y-%m-%d", time.localtime(mtime)) if mtime else "-"
            rel = _relative_path(fpath)
            if status == "active":
                status_str = _C.green("active")
            elif status == "missing":
                status_str = _C.yellow("missing")
            else:
                status_str = _C.dim(status)
            size_str = _human_size(size) if size else "-"
            print(f"  {status_str:>16}  {size_str:>8}  {ts}  {rel}")

        active = sum(1 for _, _, _, _, _, s in rows if s == "active")
        print(_C.dim(f"\n{active} file{'s' if active != 1 else ''} indexed"))
        return 0
    except Exception as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        return 2


def cmd_index(args) -> int:
    """Manually index a file or directory."""
    try:
        from lss_store import ingest_many, discover_files

        path = Path(args.path).resolve()
        if not path.exists():
            print(_C.bold_red("error:") + f" path not found: {args.path}", file=sys.stderr)
            print(f"\nMake sure the file exists and try again.", file=sys.stderr)
            return 2

        if path.is_dir():
            all_files, new_files, already = discover_files(path)
            total = len(all_files)
            new_count = len(new_files)
            scope_label = _relative_path(path)
            auto_yes = getattr(args, "yes", False)
            interactive = _is_interactive() and not auto_yes

            if new_count > 0 and interactive:
                if not _confirm_index(new_count, total, already, scope_label):
                    print(_C.dim("  skipped"), file=sys.stderr)
                    return 0

            if new_count > 0:
                # Index only the new files (not the whole dir again)
                if _is_interactive() and not args.quiet:
                    def _prog(cur, tot, p):
                        _progress_line(cur, tot, p, path)
                    uids = ingest_many(new_files, progress_cb=_prog)
                else:
                    uids = ingest_many(new_files)
                if not args.quiet:
                    print(f"Indexed {len(uids)} file{'s' if len(uids) != 1 else ''} from {scope_label}")
            else:
                if not args.quiet:
                    print(f"All {total} file{'s' if total != 1 else ''} in {scope_label} already indexed")
            return 0

        if not path.is_file():
            print(_C.bold_red("error:") + f" not a file or directory: {args.path}", file=sys.stderr)
            return 2

        ensure_indexed(path)
        if not args.quiet:
            print(f"Indexed {_relative_path(path)}")
        return 0
    except ValueError as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        if "Not a text file" in str(e):
            print(f"\nLSS only indexes text-based files (code, docs, configs, etc.)", file=sys.stderr)
        return 2
    except Exception as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        return 2


def cmd_eval(args) -> int:
    """Run the search quality evaluation harness."""
    try:
        eval_dir = Path(__file__).parent / "tests" / "evaluation"
        golden_set = eval_dir / "golden_set.json"

        if not golden_set.exists():
            print(
                _C.bold_red("error:")
                + " evaluation files not found at "
                + str(eval_dir),
                file=sys.stderr,
            )
            print(
                "Run from the lss source tree, or install dev dependencies.",
                file=sys.stderr,
            )
            return 2

        # Check dependencies
        try:
            import ranx  # noqa: F401
        except ImportError:
            print(
                _C.bold_red("error:")
                + " ranx not installed (required for eval)",
                file=sys.stderr,
            )
            print("Install with: pip install 'lss[dev]'", file=sys.stderr)
            return 2

        if not _check_openai_key():
            return 1

        # Ensure the evaluation package is importable
        import sys as _sys
        tests_dir = str(Path(__file__).parent / "tests")
        if tests_dir not in _sys.path:
            _sys.path.insert(0, tests_dir)

        from evaluation.corpus import generate_corpus, CORPUS_FILES
        from evaluation.harness import SearchEvalHarness
        from lss_store import discover_files, ingest_many

        # Create temp corpus
        import tempfile
        with tempfile.TemporaryDirectory(prefix="lss_eval_") as tmpdir:
            tmpdir = Path(tmpdir).resolve()  # resolve symlinks (macOS /var -> /private/var)
            corpus_path = tmpdir / "project"
            lss_dir = tmpdir / ".lss"
            lss_dir.mkdir()

            # Point modules at temp dir
            import lss_config as _cfg
            import lss_store as _store
            import semantic_search as _sem

            old_lss_dir = _cfg.LSS_DIR
            old_lss_db = _cfg.LSS_DB
            old_config_file = _cfg.CONFIG_FILE
            old_timeout = _sem.OAI_TIMEOUT

            _cfg.LSS_DIR = lss_dir
            _cfg.LSS_DB = lss_dir / "lss.db"
            _cfg.CONFIG_FILE = lss_dir / "config.json"
            _store.LSS_DIR = lss_dir
            _store.LSS_DB = lss_dir / "lss.db"
            _store._file_cache.clear()
            _sem.OAI_TIMEOUT = 15.0

            try:
                # Generate corpus
                print(_C.dim("generating corpus..."), file=sys.stderr)
                generate_corpus(corpus_path)

                # Index
                print(_C.dim("indexing..."), file=sys.stderr)
                all_files, new_files, _ = discover_files(corpus_path)
                ingest_many(new_files)
                print(
                    _C.dim(f"  {len(all_files)} files, {len(new_files)} indexed"),
                    file=sys.stderr,
                )

                # Run evaluation
                print(_C.dim("running evaluation (BM25 / embedding / hybrid)..."), file=sys.stderr)
                harness = SearchEvalHarness(golden_set)
                report = harness.full_evaluation(corpus_path)

                # Output
                use_json = getattr(args, "json", False)
                if use_json:
                    print(json.dumps(report.to_dict(), indent=2))
                else:
                    print(report.summary_table())
                    print(
                        _C.dim(f"  corpus: {report.corpus_files} files, "
                               f"{report.total_queries} queries")
                    )

                return 0
            finally:
                # Restore original config
                _cfg.LSS_DIR = old_lss_dir
                _cfg.LSS_DB = old_lss_db
                _cfg.CONFIG_FILE = old_config_file
                _store.LSS_DIR = old_lss_dir
                _store.LSS_DB = old_lss_db
                _store._file_cache.clear()
                _sem.OAI_TIMEOUT = old_timeout

    except KeyboardInterrupt:
        print("\n" + _C.dim("  interrupted"), file=sys.stderr)
        return 130
    except Exception as e:
        print(_C.bold_red("error:") + f" {e}", file=sys.stderr)
        if getattr(args, "debug", False):
            import traceback
            traceback.print_exc()
        return 2


def cmd_version(_args) -> int:
    print(f"lss {__version__}")
    return 0


# ── Status ───────────────────────────────────────────────────────────────────

def cmd_status(args) -> int:
    """Show LSS status: config, watched paths, exclusions, DB stats."""
    cfg = lss_config.load_config()

    db_path = Path(get_db_path())
    db_exists = db_path.exists()

    # Header
    print(_C.bold(f"lss {__version__}"))
    print(f"  {_C.dim('data dir')}     {lss_config.LSS_DIR}")
    print(f"  {_C.dim('config')}       {lss_config.CONFIG_FILE}")

    if db_exists:
        total = sum(
            f.stat().st_size
            for f in db_path.parent.glob(db_path.name + "*")
            if f.is_file()
        )
        print(f"  {_C.dim('database')}     {db_path}  {_C.dim('(' + _human_size(total) + ')')}")
    else:
        print(f"  {_C.dim('database')}     {_C.dim('not created yet')}")

    # Watch paths
    watch = cfg.get("watch_paths", [])
    print()
    if watch:
        print(f"  {_C.bold('watched paths')} ({len(watch)}):")
        for wp in watch:
            exists = Path(wp).exists()
            if exists:
                print(f"    {_C.green(wp)}")
            else:
                print(f"    {wp}  {_C.yellow('[missing]')}")
    else:
        print(f"  {_C.dim('watched paths: none')}")
        print(f"    {_C.dim('add with:')} lss watch add ~/Documents")

    # Exclusion patterns
    excludes = cfg.get("exclude_patterns", [])
    print()
    if excludes:
        print(f"  {_C.bold('exclusion patterns')} ({len(excludes)}):")
        for pat in excludes:
            print(f"    {_C.cyan(pat)}")
    else:
        print(f"  {_C.dim('exclusion patterns: none')}")

    # API key status
    has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    print()
    if has_key:
        print(f"  {_C.dim('api key')}      {_C.green('set')}")
    else:
        print(f"  {_C.dim('api key')}      {_C.yellow('not set')}  {_C.dim('(required for search)')}")

    # DB stats
    if db_exists:
        try:
            con = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
            con.execute("PRAGMA busy_timeout=30000")
            total_files = con.execute(
                "SELECT COUNT(*) FROM files WHERE status='active'"
            ).fetchone()[0]
            total_chunks = con.execute("SELECT COUNT(*) FROM fts").fetchone()[0]
            total_embeds = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

            row = con.execute(
                "SELECT MAX(indexed_at) FROM files WHERE status='active'"
            ).fetchone()
            last_indexed = row[0] if row and row[0] else None
            con.close()

            print()
            print(f"  {_C.dim('indexed files')}  {_C.bold(str(total_files))}")
            print(f"  {_C.dim('text chunks')}    {total_chunks}")
            print(f"  {_C.dim('embeddings')}     {total_embeds}")
            if last_indexed:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_indexed))
                print(f"  {_C.dim('last indexed')}   {ts} {_C.dim('(' + _time_ago(last_indexed) + ')')}")
        except Exception:
            pass

    print()
    return 0


# ── Watch / Exclude ──────────────────────────────────────────────────────────

def cmd_watch(args) -> int:
    cfg = lss_config.load_config()
    action = args.watch_action

    if action == "list":
        paths = cfg.get("watch_paths", [])
        if not paths:
            print(_C.dim("No watched paths."))
            print(f"  {_C.dim('add with:')} lss watch add <path>")
            return 0
        for p in paths:
            exists = Path(p).exists()
            if exists:
                print(f"  {_C.green(p)}")
            else:
                print(f"  {p}  {_C.yellow('[missing]')}")
        return 0

    if action == "add":
        resolved = str(Path(args.path).resolve())
        paths = cfg.get("watch_paths", [])
        if resolved in paths:
            print(f"Already watching: {resolved}")
            return 0
        paths.append(resolved)
        cfg["watch_paths"] = paths
        lss_config.save_config(cfg)
        print(f"Added: {_C.green(resolved)}")
        print(_C.dim("  Start the watcher with: lss-sync"))
        return 0

    if action == "remove":
        resolved = str(Path(args.path).resolve())
        paths = cfg.get("watch_paths", [])
        if resolved in paths:
            paths.remove(resolved)
        elif args.path in paths:
            paths.remove(args.path)
        else:
            print(f"Not watching: {resolved}")
            return 1
        cfg["watch_paths"] = paths
        lss_config.save_config(cfg)
        print(f"Removed: {resolved}")
        return 0

    return 2


def cmd_exclude(args) -> int:
    cfg = lss_config.load_config()
    action = args.exclude_action

    if action == "list":
        patterns = cfg.get("exclude_patterns", [])
        if not patterns:
            print(_C.dim("No exclusion patterns."))
            print(f"  {_C.dim('add with:')} lss exclude add <pattern>")
            return 0
        for p in patterns:
            print(f"  {_C.cyan(p)}")
        return 0

    if action == "add":
        pattern = args.pattern
        patterns = cfg.get("exclude_patterns", [])
        if pattern in patterns:
            print(f"Already excluded: {pattern}")
            return 0
        patterns.append(pattern)
        cfg["exclude_patterns"] = patterns
        lss_config.save_config(cfg)
        print(f"Added exclusion: {_C.cyan(pattern)}")
        return 0

    if action == "remove":
        pattern = args.pattern
        patterns = cfg.get("exclude_patterns", [])
        if pattern not in patterns:
            print(f"Not excluded: {pattern}")
            return 1
        patterns.remove(pattern)
        cfg["exclude_patterns"] = patterns
        lss_config.save_config(cfg)
        print(f"Removed exclusion: {pattern}")
        return 0

    return 2


# ── Parser ───────────────────────────────────────────────────────────────────

_KNOWN_SUBCOMMANDS = {
    "search", "status", "watch", "exclude", "sweep",
    "db-path", "ls", "index", "version", "eval",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lss",
        description="Local Semantic Search — find anything in your files",
        usage="lss <query> [path] [options]\n       lss <command> [args]",
    )
    p.add_argument("-v", "--version", action="version", version=f"lss {__version__}")
    sub = p.add_subparsers(dest="cmd")

    # ── search (the default, also available as explicit subcommand) ──
    p_s = sub.add_parser("search", help="search files (this is the default)")
    p_s.add_argument("queries", nargs="*", help="search queries")
    p_s.add_argument("-p", "--path", default=None,
                     help="path to search (default: current directory)")
    p_s.add_argument("-Q", "--qfile",
                     help="file with one query per line, or '-' for stdin")
    p_s.add_argument("-k", "--limit", type=int, default=5,
                     help="max results per query (default: 5)")
    p_s.add_argument("--json", action="store_true", help="JSON output")
    p_s.add_argument("--no-index", action="store_true",
                     help="skip auto-indexing, only search already-indexed files")
    # Keep old flag name for backwards compat
    p_s.add_argument("--indexed-only", action="store_true",
                     help=argparse.SUPPRESS)
    p_s.add_argument("-y", "--yes", action="store_true",
                     help="skip confirmation prompt before indexing")
    p_s.add_argument("--no-color", action="store_true", help="disable colors")
    p_s.add_argument("--debug", action="store_true", help="debug output")
    p_s.set_defaults(func=cmd_search)

    # ── status ──
    p_st = sub.add_parser("status", help="show config, watched paths, DB stats")
    p_st.add_argument("--debug", action="store_true")
    p_st.set_defaults(func=cmd_status)

    # ── watch ──
    p_w = sub.add_parser("watch", help="manage watched directories")
    p_w_sub = p_w.add_subparsers(dest="watch_action", required=True)
    p_w_sub.add_parser("list", help="list watched paths")
    p_w_add = p_w_sub.add_parser("add", help="add a watch path")
    p_w_add.add_argument("path", help="directory to watch")
    p_w_rm = p_w_sub.add_parser("remove", help="remove a watch path")
    p_w_rm.add_argument("path", help="directory to stop watching")
    p_w.add_argument("--debug", action="store_true")
    p_w.set_defaults(func=cmd_watch)

    # ── exclude ──
    p_e = sub.add_parser("exclude", help="manage exclusion patterns")
    p_e_sub = p_e.add_subparsers(dest="exclude_action", required=True)
    p_e_sub.add_parser("list", help="list exclusion patterns")
    p_e_add = p_e_sub.add_parser("add", help="add an exclusion pattern")
    p_e_add.add_argument("pattern", help="pattern (glob *, dir name, or path with /)")
    p_e_rm = p_e_sub.add_parser("remove", help="remove an exclusion pattern")
    p_e_rm.add_argument("pattern", help="pattern to remove")
    p_e.add_argument("--debug", action="store_true")
    p_e.set_defaults(func=cmd_exclude)

    # ── sweep ──
    p_sw = sub.add_parser("sweep", help="cleanup / remove indexed data")
    p_sw.add_argument("--remove", nargs="+", metavar="PATH",
                      help="remove specific files from index")
    p_sw.add_argument("--clear-all", action="store_true",
                      help="delete DB and recreate empty")
    p_sw.add_argument("--clear-embeddings", type=int, metavar="DAYS",
                      help="delete embeddings older than DAYS (0 = all)")
    p_sw.add_argument("--retention-days", type=int, default=30,
                      help="sweep retention window (default: 30)")
    p_sw.add_argument("--no-optimize", action="store_true",
                      help="skip FTS optimize + WAL checkpoint")
    p_sw.add_argument("--debug", action="store_true")
    p_sw.set_defaults(func=cmd_sweep)

    # ── utility ──
    p_dp = sub.add_parser("db-path", help="print database path")
    p_dp.add_argument("--debug", action="store_true")
    p_dp.set_defaults(func=cmd_dbpath)

    p_ls = sub.add_parser("ls", help="list indexed files")
    p_ls.add_argument("--debug", action="store_true")
    p_ls.set_defaults(func=cmd_ls)

    p_idx = sub.add_parser("index", help="index a file or directory")
    p_idx.add_argument("path", help="file or directory to index")
    p_idx.add_argument("-q", "--quiet", action="store_true")
    p_idx.add_argument("-y", "--yes", action="store_true",
                       help="skip confirmation prompt before indexing")
    p_idx.add_argument("--debug", action="store_true")
    p_idx.set_defaults(func=cmd_index)

    p_eval = sub.add_parser("eval", help="run search quality evaluation")
    p_eval.add_argument("--json", action="store_true", help="JSON output")
    p_eval.add_argument("--debug", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_v = sub.add_parser("version", help="show version")
    p_v.set_defaults(func=cmd_version)

    return p


def main(argv=None) -> int:
    """Entry point with smart default-to-search behavior.

    If the first argument is not a known subcommand, treat the entire
    argv as an implicit ``lss search ...`` invocation.  This lets users type
    ``lss "my query"`` instead of ``lss search "my query"``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Smart routing: if the first arg isn't a known subcommand, assume search
    if argv and argv[0] not in _KNOWN_SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["search"] + list(argv)

    parser = build_parser()

    # Handle empty args
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    # Set global debug flag
    debug_enabled = getattr(args, "debug", False)
    lss_config.set_debug(debug_enabled)

    if debug_enabled:
        print(f"[DEBUG] Debug mode enabled, LSS_DIR={lss_config.LSS_DIR}")

    # Respect --no-color flag
    if getattr(args, "no_color", False):
        _C.set_enabled(False)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
