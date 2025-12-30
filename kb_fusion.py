#!/usr/bin/env python3
import argparse, sys, json, os, sqlite3, time
from typing import List

__version__ = "0.1.2"

# Set debug BEFORE importing other modules
import kb_config

# flat imports
from kb_store import (
    ensure_indexed,
    get_db_path,
    remove_files,
    sweep as sweep_default,
    clear_all,
    clear_embeddings,
)


# ---------- helpers ----------
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


# ---------- commands ----------
def cmd_search(args) -> int:
    from pathlib import Path
    from kb_store import ingest_many, get_file_uid, _is_text_file
    from semantic_search import semantic_search
    
    qs = _read_queries(args)
    if not qs:
        print("[search] provide queries as args or via -Q/--qfile (use '-' for stdin)",
              file=sys.stderr)
        return 2
    
    try:
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"[search] ERROR path not found: {args.path}", file=sys.stderr)
            return 2
        
        unindexed_files = []
        
        # Handle indexed-only mode
        if args.indexed_only:
            if path.is_file():
                file_uid = get_file_uid(path)
                if not file_uid:
                    # File not indexed - return special result
                    unindexed_files.append(str(path))
            elif path.is_dir():
                # Find all text files in directory
                all_files = []
                for file_path in path.rglob("*"):
                    if file_path.is_file() and _is_text_file(file_path):
                        all_files.append(file_path)
                
                # Check which are not indexed
                for file_path in all_files:
                    if not get_file_uid(file_path):
                        unindexed_files.append(str(file_path))
        else:
            # Normal mode: auto-index
            if path.is_file():
                ensure_indexed(path)
            elif path.is_dir():
                ingest_many(path)
            else:
                print(f"[search] ERROR path must be a file or directory: {args.path}", file=sys.stderr)
                return 2
        
        t0 = time.time()
        batches = semantic_search(str(path), qs, indexed_only=args.indexed_only)
        dt = time.time() - t0

        if args.json:
            payload = []
            for q, hits in zip(qs, batches):
                # Clean hits: remove chunk_id and file_uid, keep essential fields
                clean_hits = []
                for h in (hits[: args.limit] if hits else []):
                    clean_hit = {
                        "file_path": h.get("file_path"),
                        "score": h.get("score"),
                        "snippet": h.get("snippet"),
                        "rank_stage": h.get("rank_stage"),
                        "indexed_at": h.get("indexed_at")
                    }
                    clean_hits.append(clean_hit)
                
                query_result = {"query": q, "hits": clean_hits}
                if unindexed_files:
                    query_result["unindexed_files"] = unindexed_files
                payload.append(query_result)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        # human output
        scope_type = "folder" if path.is_dir() else "file"
        mode_note = " (indexed-only)" if args.indexed_only else ""
        print(f"# {scope_type}: {args.path}{mode_note}")
        print(f"# queries: {len(qs)} | top-k: {args.limit} | {dt:.2f}s")
        
        if unindexed_files:
            print(f"# unindexed files: {len(unindexed_files)}")
            for uf in unindexed_files[:5]:  # Show first 5
                print(f"#   - {uf}")
            if len(unindexed_files) > 5:
                print(f"#   ... and {len(unindexed_files) - 5} more")
        print()
        
        for i, (q, hits) in enumerate(zip(qs, batches), 1):
            print(f"Q{i}: {q}")
            if not hits:
                print("  (no hits)")
                continue
            for j, h in enumerate(hits[: args.limit], 1):
                score = f"{h.get('score', 0.0):.3f}"
                stage = h.get("rank_stage", "")
                indexed_at = h.get("indexed_at")
                indexed_str = ""
                if indexed_at:
                    indexed_str = f" [indexed: {time.strftime('%Y-%m-%d %H:%M', time.localtime(indexed_at))}]"
                snippet = (h.get("snippet", "") or "").replace("\n", " ").strip()
                print(f"  {j:>2}. {score} [{stage}]{indexed_str}\n      {snippet}")
            print()
        return 0
    except Exception as e:
        print(f"[search] ERROR {e}", file=sys.stderr)
        return 2


def cmd_sweep(args) -> int:
    """Maintenance:
       --remove PATH [PATH ...]     remove specific indexed files by absolute path
       --clear-all                  delete the DB files and recreate empty schema
       --clear-embeddings DAYS      delete embeddings older than DAYS (0 = delete all)
       --no-optimize                skip FTS optimize & WAL checkpoint (avoids lock)
       --retention-days N           generic sweep embedding retention window (default 30)
    """
    try:
        if args.clear_all:
            clear_all()
            print("[sweep] cleared database (schema recreated)")

        elif args.remove:
            # uses kb_store.remove_files (handles fts/files/embeddings)
            n = remove_files(args.remove)
            if n == 0:
                print("[sweep] nothing removed (no matching paths)")
            else:
                print(f"[sweep] removed {n} file(s)")

        elif args.clear_embeddings is not None:
            days = None if args.clear_embeddings == 0 else args.clear_embeddings
            clear_embeddings(days=days)
            print(f"[sweep] cleared embeddings ({'ALL' if days is None else f'older than {days}d'})")

        else:
            # marks missing, prunes orphaned/old embeddings
            sweep_default(retention_days=args.retention_days)

        # optional housekeeping
        if not args.no_optimize:
            try:
                con = sqlite3.connect(get_db_path(), check_same_thread=False, timeout=5)
                cur = con.cursor()
                cur.execute("INSERT INTO fts(fts) VALUES('optimize')")
                cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.commit()
                con.close()
            except Exception as e:
                print(f"[sweep] WARN optimize skipped: {e}", file=sys.stderr)

        return 0
    except Exception as e:
        print(f"[sweep] ERROR {e}", file=sys.stderr)
        return 2


def cmd_dbpath(_args) -> int:
    try:
        print(get_db_path())
        return 0
    except Exception as e:
        print(f"[db-path] ERROR {e}", file=sys.stderr)
        return 2


def cmd_ls(_args) -> int:
    """List what’s indexed (helps debug ‘not indexed’)."""
    try:
        con = sqlite3.connect(get_db_path(), check_same_thread=False, timeout=30)
        cur = con.cursor()
        rows = cur.execute(
            "SELECT file_uid, path, size, mtime, version, status "
            "FROM files ORDER BY status DESC, path"
        ).fetchall()
        con.close()
        if not rows:
            print("(empty)")
            return 0
        for uid, path, size, mtime, version, status in rows:
            ts = time.strftime("%Y-%m-%d", time.localtime(mtime)) if mtime else "-"
            print(f"{status:8} {size:9} {ts}  {uid}  {path}")
        return 0
    except Exception as e:
        print(f"[ls] ERROR {e}", file=sys.stderr)
        return 2


def cmd_index(args) -> int:
    """Manually index a single file."""
    from pathlib import Path
    
    try:
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"[index] ERROR path not found: {args.path}", file=sys.stderr)
            return 2
        
        if not path.is_file():
            print(f"[index] ERROR path must be a file: {args.path}", file=sys.stderr)
            return 2
        
        ensure_indexed(path)
        if not args.quiet:
            print(f"[index] indexed: {path}")
        return 0
    except Exception as e:
        print(f"[index] ERROR {e}", file=sys.stderr)
        return 2


def cmd_version(_args) -> int:
    """Print version information."""
    print(f"kb-fusion {__version__}")
    return 0


# ---------- parser / entry ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kb", description="kb-fusion CLI")
    p.add_argument("-v", "--version", action="version", version=f"kb-fusion {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # search (auto-ingests the file/folder on first run)
    p_s = sub.add_parser("search", help="search within a file or folder")
    p_s.add_argument("path", help="path to the file or folder to search")
    p_s.add_argument("queries", nargs="*", help="one or more query strings")
    p_s.add_argument("-Q", "--qfile",
                     help="file with one query per line, or '-' to read queries from stdin")
    p_s.add_argument("-k", "--limit", type=int, default=10, help="max results (default: 10)")
    p_s.add_argument("--json", action="store_true", help="print raw JSON")
    p_s.add_argument("--indexed-only", action="store_true",
                     help="only search indexed files, skip auto-indexing (reports unindexed files)")
    p_s.add_argument("--debug", action="store_true", help="enable debug output")
    p_s.set_defaults(func=cmd_search)

    # sweep / maintenance
    p_sw = sub.add_parser("sweep", help="cleanup / remove indexed data")
    p_sw.add_argument("--remove", nargs="+", metavar="PATH",
                      help="remove one or more indexed files by absolute path")
    p_sw.add_argument("--clear-all", action="store_true",
                      help="delete DB files and recreate empty schema")
    p_sw.add_argument("--clear-embeddings", type=int, metavar="DAYS",
                      help="delete embeddings older than DAYS; 0 deletes ALL")
    p_sw.add_argument("--retention-days", type=int, default=30,
                      help="embedding retention for generic sweep (default: 30)")
    p_sw.add_argument("--no-optimize", action="store_true",
                      help="skip FTS optimize and WAL checkpoint")
    p_sw.add_argument("--debug", action="store_true", help="enable debug output")
    p_sw.set_defaults(func=cmd_sweep)

    # utility
    p_dp = sub.add_parser("db-path", help="print KB database path")
    p_dp.add_argument("--debug", action="store_true", help="enable debug output")
    p_dp.set_defaults(func=cmd_dbpath)

    p_ls = sub.add_parser("ls", help="list indexed files")
    p_ls.add_argument("--debug", action="store_true", help="enable debug output")
    p_ls.set_defaults(func=cmd_ls)

    p_idx = sub.add_parser("index", help="manually index a single file")
    p_idx.add_argument("path", help="path to the file to index")
    p_idx.add_argument("-q", "--quiet", action="store_true", help="suppress success messages")
    p_idx.add_argument("--debug", action="store_true", help="enable debug output")
    p_idx.set_defaults(func=cmd_index)

    p_v = sub.add_parser("version", help="show version information")
    p_v.set_defaults(func=cmd_version)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Set global debug flag
    debug_enabled = getattr(args, 'debug', False)
    kb_config.set_debug(debug_enabled)
    
    if debug_enabled:
        print(f"[DEBUG] Debug mode enabled, KB_DIR={kb_config.KB_DIR}")
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
