import os, time, sqlite3, hashlib, json, re, fnmatch
from pathlib import Path
from collections import OrderedDict

from lss_config import LSS_DIR, LSS_DB, VERSION_KEY
import lss_config

# ── Default exclusions ────────────────────────────────────────────────────────
#
# These lists prevent indexing junk that wastes time, disk, and API calls.
# Users can add more via `lss exclude add <pattern>`.

EXCLUDED_DIRS = {
    # Version control
    '.git', '.svn', '.hg', '.bzr', '.fossil',
    # Dependencies / package managers
    'node_modules', 'vendor', 'bower_components', '.pnpm', '.yarn',
    'jspm_packages', 'web_modules',
    # Python
    '__pycache__', '.venv', 'venv', 'env', '.env', '.tox',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', '.pytype',
    'site-packages', '.eggs', '*.egg-info',
    # Ruby
    '.bundle', 'vendor/bundle',
    # Rust
    'target',
    # Go
    'vendor',
    # Java / JVM
    '.gradle', '.mvn', '.m2',
    # .NET
    'bin', 'obj', 'packages',
    # Build outputs
    'dist', 'build', '_build', 'out', 'output', '.output',
    '.next', '.nuxt', '.svelte-kit', '.vercel', '.netlify',
    '.parcel-cache', '.webpack', '.rollup.cache',
    # IDEs / editors
    '.idea', '.vscode', '.vs', '.eclipse', '.settings',
    '.project', '.classpath', '.factorypath',
    # OS junk
    '.DS_Store', 'Thumbs.db', '.Spotlight-V100', '.Trashes',
    'ehthumbs.db', 'Desktop.ini', '$RECYCLE.BIN',
    # Caches / temp
    '.cache', '.tmp', 'tmp', 'temp', '.temp',
    '.sass-cache', '.eslintcache', '.stylelintcache',
    # Test / coverage
    'coverage', '.nyc_output', 'htmlcov', '.coverage',
    '__snapshots__', '.jest',
    # Containers / infra
    '.terraform', '.vagrant', '.docker',
    # Monorepo / tooling
    '.turbo', '.nx', '.rush',
    # Data / logs (usually large and noisy)
    'logs', 'log',
    # LSS's own data
    '.lss',
}

# File extensions that are ALWAYS binary — skip without opening the file.
# This is a fast-path optimisation: _is_text_file won't even read bytes.
BINARY_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.icns',
    '.tiff', '.tif', '.webp', '.avif', '.heic', '.heif',
    '.psd', '.ai', '.eps', '.raw', '.cr2', '.nef', '.svg',
    # Video
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp',
    # Audio
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a',
    '.opus', '.mid', '.midi',
    # Archives / compressed
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    '.tgz', '.tar.gz', '.tar.bz2', '.tar.xz', '.zst',
    '.dmg', '.iso', '.img',
    # Compiled / bytecode
    '.pyc', '.pyo', '.class', '.o', '.obj', '.so', '.dylib',
    '.dll', '.exe', '.bin', '.elf', '.wasm',
    '.a', '.lib', '.ko',
    # Documents (binary formats)
    '.pdf',  # we handle PDF separately via _extract_pdf_text
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.odt', '.ods', '.odp', '.pages', '.numbers', '.key',
    # Fonts
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    # Database files
    '.db', '.sqlite', '.sqlite3', '.mdb', '.accdb',
    '.db-wal', '.db-shm',
    # Package files
    '.deb', '.rpm', '.apk', '.msi', '.pkg', '.snap',
    # Serialized / binary data
    '.pkl', '.pickle', '.npy', '.npz', '.h5', '.hdf5',
    '.parquet', '.avro', '.arrow', '.feather',
    '.pb', '.onnx', '.pt', '.pth', '.safetensors',
    '.dat', '.bin',
    # Certificates / keys (also sensitive)
    '.p12', '.pfx', '.pem', '.der', '.cer', '.crt', '.key',
    # Misc binary
    '.blend', '.fbx', '.glb', '.gltf', '.usd', '.usda',
    '.swf', '.fla',
    # Lock files (text but useless for search)
    '.lock',
    # Log files (constantly changing, mostly noise)
    '.log',
    # Source maps (huge, generated)
    '.map',
    # Minified files
    '.min.js', '.min.css',
}

# File NAMES (exact match, case-sensitive) to always skip.
EXCLUDED_FILES = {
    # Lock files
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
    'Gemfile.lock', 'Pipfile.lock', 'poetry.lock', 'uv.lock',
    'composer.lock', 'Cargo.lock', 'go.sum', 'flake.lock',
    # Generated
    '.gitattributes', '.gitmodules',
    # Environment / secrets
    '.env', '.env.local', '.env.production', '.env.development',
    '.env.staging', '.env.test',
    # OS
    '.DS_Store', 'Thumbs.db', 'desktop.ini',
    # npm
    '.npmrc', '.npmignore',
    # Python
    'pip-log.txt', 'pip-delete-this-directory.txt',
}

# Maximum file size to index (skip huge files that blow up memory/API costs)
MAX_FILE_SIZE = int(os.environ.get("LSS_MAX_FILE_SIZE", 2 * 1024 * 1024))  # 2 MB default

# ── LRU cache ────────────────────────────────────────────────────────────────

_file_cache = OrderedDict()
_cache_limit = 8192  # large enough for big directories

def _cache_put(key, value):
    """Add to LRU cache, evicting oldest if over limit."""
    _file_cache[key] = value
    _file_cache.move_to_end(key)
    if len(_file_cache) > _cache_limit:
        _file_cache.popitem(last=False)

# ── Hashing ──────────────────────────────────────────────────────────────────

def _content_sig(file_path):
    """Content-based signature (path-agnostic)"""
    p = Path(file_path)
    size = p.stat().st_size

    if size <= 64 * 1024 * 1024:  # ≤64MB: full hash
        return hashlib.md5(p.read_bytes()).hexdigest()

    with open(p, 'rb') as f:
        first = f.read(65536)
        f.seek(size // 2)
        middle = f.read(65536)
        f.seek(max(0, size - 65536))
        last = f.read(65536)

    hasher = hashlib.md5()
    hasher.update(first + middle + last + str(size).encode())
    return hasher.hexdigest()

def _text_hash(text):
    """Hash for text chunks - 16-byte binary MD5 for space efficiency"""
    return hashlib.md5(text.encode('utf-8')).digest()

def _get_file_cache_key(path):
    """Fast cache key for file stats"""
    stat = path.stat()
    return (str(path), stat.st_size, int(stat.st_mtime), VERSION_KEY)

def _check_file_cached(file_path):
    """Fast check if file is already indexed - avoids expensive hashing"""
    path = Path(file_path).resolve()

    cache_key = _get_file_cache_key(path)
    if cache_key in _file_cache:
        _file_cache.move_to_end(cache_key)
        return _file_cache[cache_key]

    con = _init_db()
    cur = con.cursor()
    try:
        stat = path.stat()
        row = cur.execute(
            "SELECT file_uid, content_sig FROM files WHERE path = ? AND size = ? AND mtime = ? AND version = ?",
            (str(path), stat.st_size, stat.st_mtime, VERSION_KEY)
        ).fetchone()
        if row:
            _cache_put(cache_key, row[0])
            return row[0]
        return None
    finally:
        con.close()

# ── Database ─────────────────────────────────────────────────────────────────

def _init_db():
    """Initialize LSS database with optimal pragmas."""
    LSS_DIR.mkdir(exist_ok=True)

    con = sqlite3.connect(str(LSS_DB), check_same_thread=False, timeout=30)
    con.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA mmap_size=268435456;
        PRAGMA cache_size=-200000;
        PRAGMA busy_timeout=30000;
        PRAGMA wal_autocheckpoint=4000;
    """)

    con.execute("""CREATE TABLE IF NOT EXISTS files (
        file_uid TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        size INTEGER,
        mtime REAL,
        content_sig TEXT NOT NULL,
        version TEXT NOT NULL,
        indexed_at REAL,
        status TEXT DEFAULT 'active'
    )""")
    # Index on path for fast lookups in discover_files / _check_file_cached
    con.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")

    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS fts
                   USING fts5(id UNINDEXED, text, file_uid UNINDEXED, file_path UNINDEXED, text_hash UNINDEXED,
                              tokenize='porter', prefix='2')""")

    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS fts_vocab
                   USING fts5vocab(fts, row)""")

    con.execute("""CREATE TABLE IF NOT EXISTS embeddings (
        text_hash BLOB,
        model TEXT,
        dim INTEGER,
        version TEXT,
        vector BLOB,
        created REAL DEFAULT (unixepoch()),
        PRIMARY KEY (text_hash, model, dim, version)
    )""")

    con.commit()
    return con

# ── Text detection ───────────────────────────────────────────────────────────

def _is_text_file(file_path):
    """Detect if a file contains text content.

    Fast-path (no I/O): extension / name / size checks.
    Slow-path: reads first 8 KB for null-byte / encoding detection.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    if ext in BINARY_EXTENSIONS:
        return False
    if name in EXCLUDED_FILES:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0 or size > MAX_FILE_SIZE:
        return False
    if ext == '.pdf':
        return True

    try:
        with open(path, 'rb') as f:
            chunk = f.read(8192)
        if not chunk:
            return False
        if b'\0' in chunk:
            return False
        try:
            chunk.decode('utf-8')
            return True
        except UnicodeDecodeError:
            pass
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                chunk.decode(encoding)
                printable_ratio = sum(1 for b in chunk if 32 <= b <= 126 or b in [9, 10, 13]) / len(chunk)
                return printable_ratio > 0.7
            except UnicodeDecodeError:
                continue
        return False
    except Exception:
        return False

# ── Text extraction ──────────────────────────────────────────────────────────

def _extract_pdf_text(file_path):
    """Extract text from PDF files using PyPDF2 (lightweight, pure Python)"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        raise RuntimeError("PDF support requires PyPDF2. Install with: pip install pypdf2")

def _extract_text(file_path):
    """Extract text from supported file types"""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == '.pdf':
        return _extract_pdf_text(file_path)

    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode text from {file_path}")

    if ext == '.json':
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return " ".join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
            elif isinstance(data, list):
                return " ".join(str(item) for item in data if isinstance(item, (str, int, float)))
            else:
                return str(data)
        except json.JSONDecodeError:
            pass
    elif ext == '.jsonl':
        try:
            texts = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if 'text' in obj:
                    texts.append(obj['text'])
                elif 'title' in obj and 'text' in obj:
                    texts.append(f"{obj['title']} {obj['text']}")
                else:
                    texts.append(" ".join(str(v) for v in obj.values() if isinstance(v, (str, int, float))))
            return "\n".join(texts)
        except json.JSONDecodeError:
            pass

    if ext == '.csv':
        try:
            import csv, io
            csvfile = io.StringIO(text)
            reader = csv.reader(csvfile)
            rows = list(reader)
            if rows:
                result = " ".join(rows[0])
                for row in rows[1:min(100, len(rows))]:
                    result += " " + " ".join(row)
                return result
        except Exception:
            pass

    return text

def _normalize_text(text):
    """Normalize text"""
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', ' ', text).strip()

def _span_chunk(text, words_per_span=220, stride=200):
    """Chunk text into overlapping spans of words."""
    words = text.split()
    if not words:
        return []

    spans = []
    start = 0
    while start < len(words):
        end = min(start + words_per_span, len(words))
        chunk = ' '.join(words[start:end])
        spans.append(("simple", chunk))
        if end >= len(words):
            break
        start += stride

    return spans

# ── File walking (os.walk with directory pruning) ────────────────────────────

def _walk_text_files(base_path, extra_exclude_patterns=None):
    """Walk a directory tree, pruning excluded dirs IN-PLACE (never enters them).

    Uses os.walk instead of rglob — on a Desktop with 120K files and 117K in
    node_modules, this is ~6x faster because pruned dirs are never traversed.

    Returns an iterator of Path objects (text files only).
    """
    base = str(Path(base_path).resolve())

    # Pre-compute user exclusion patterns (load config ONCE, not per-file)
    cfg_patterns = list(extra_exclude_patterns or [])
    cfg = lss_config.load_config()
    cfg_patterns.extend(cfg.get("exclude_patterns", []))

    # Split patterns into dir-name patterns and file-glob patterns
    dir_excludes = set(EXCLUDED_DIRS)
    file_globs = []
    for pat in cfg_patterns:
        if '/' not in pat and '*' not in pat and '?' not in pat:
            dir_excludes.add(pat)  # bare name → treat as dir exclusion too
        file_globs.append(pat)

    for root, dirs, files in os.walk(base):
        # Prune excluded directories in-place — os.walk won't descend into them
        dirs[:] = [d for d in dirs if d not in dir_excludes]

        for name in files:
            # Fast-path: name and extension checks (no I/O)
            if name in EXCLUDED_FILES:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in BINARY_EXTENSIONS:
                continue

            # Check user-configured file glob patterns
            if file_globs:
                skip = False
                for pat in file_globs:
                    if fnmatch.fnmatch(name, pat):
                        skip = True
                        break
                if skip:
                    continue

            full = os.path.join(root, name)

            # Size check (no open, just stat)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size == 0 or size > MAX_FILE_SIZE:
                continue

            # PDF special case
            if ext == '.pdf':
                yield Path(full)
                continue

            # Slow-path: byte-level text detection
            try:
                with open(full, 'rb') as f:
                    chunk = f.read(8192)
                if not chunk:
                    continue
                if b'\0' in chunk:
                    continue
                try:
                    chunk.decode('utf-8')
                    yield Path(full)
                    continue
                except UnicodeDecodeError:
                    pass
                for enc in ('latin-1', 'cp1252', 'iso-8859-1'):
                    try:
                        chunk.decode(enc)
                        ratio = sum(1 for b in chunk if 32 <= b <= 126 or b in (9, 10, 13)) / len(chunk)
                        if ratio > 0.7:
                            yield Path(full)
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception:
                continue

# ── Core indexing API ────────────────────────────────────────────────────────

def ensure_indexed(file_path, force_reindex=False):
    """Ensure file is indexed - main API with fast path"""
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not _is_text_file(path):
        raise ValueError(f"Not a text file: {path.name}")

    if not force_reindex:
        cached_uid = _check_file_cached(path)
        if cached_uid:
            return cached_uid

    return _do_index(path)

def _do_index(path, con=None):
    """Actually perform the indexing work.

    If `con` is provided, reuses that connection (batch mode).
    If `con` is None, opens and closes its own connection (single-file mode).
    """
    own_con = con is None
    if own_con:
        con = _init_db()
    cur = con.cursor()

    try:
        stat = path.stat()
        size, mtime = stat.st_size, stat.st_mtime
        content_sig = _content_sig(path)
        file_uid = f"f_{content_sig}"

        row = cur.execute(
            "SELECT content_sig, version FROM files WHERE file_uid = ?",
            (file_uid,)
        ).fetchone()

        if row and row[0] == content_sig and row[1] == VERSION_KEY:
            cur.execute("UPDATE files SET path = ? WHERE file_uid = ?", (str(path), file_uid))
            if own_con:
                con.commit()
            _cache_put(_get_file_cache_key(path), file_uid)
            return file_uid

        if lss_config.DEBUG:
            print(f"Indexing: {path.name}")

        # Clean up old entries for this path (handles content changes)
        old_entries = cur.execute(
            "SELECT file_uid FROM files WHERE path = ? AND file_uid != ?",
            (str(path), file_uid)
        ).fetchall()
        for (old_uid,) in old_entries:
            old_hashes = cur.execute(
                "SELECT text_hash FROM fts WHERE file_uid = ?", (old_uid,)
            ).fetchall()
            cur.execute("DELETE FROM fts WHERE file_uid = ?", (old_uid,))
            if old_hashes:
                placeholders = ','.join(['?' for _ in old_hashes])
                cur.execute(f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})",
                           [h[0] for h in old_hashes])
            cur.execute("DELETE FROM files WHERE file_uid = ?", (old_uid,))

        raw_text = _extract_text(path)
        text = _normalize_text(raw_text)
        spans = _span_chunk(text)

        existing_hashes = set()
        if row:
            existing_rows = cur.execute(
                "SELECT text_hash FROM fts WHERE file_uid = ?", (file_uid,)
            ).fetchall()
            existing_hashes = {r[0] for r in existing_rows}

        new_spans = []
        new_hashes = set()
        for i, (chunk_type, span_text) in enumerate(spans):
            span_id = f"{file_uid}::{i}" if len(spans) > 1 else file_uid
            text_hash_val = _text_hash(span_text)
            new_spans.append((span_id, span_text, file_uid, str(path), text_hash_val))
            new_hashes.add(text_hash_val)

        stale_hashes = existing_hashes - new_hashes
        fresh_hashes = new_hashes - existing_hashes

        if stale_hashes:
            placeholders = ','.join(['?' for _ in stale_hashes])
            cur.execute(f"DELETE FROM fts WHERE file_uid = ? AND text_hash IN ({placeholders})",
                       [file_uid] + list(stale_hashes))
            cur.execute(f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})",
                       list(stale_hashes))

        fresh_spans = [s for s in new_spans if s[4] in fresh_hashes]
        if fresh_spans:
            cur.executemany("INSERT INTO fts(id, text, file_uid, file_path, text_hash) VALUES(?,?,?,?,?)",
                           fresh_spans)

        cur.execute("""INSERT OR REPLACE INTO files
                      (file_uid, path, size, mtime, content_sig, version, indexed_at, status)
                      VALUES (?,?,?,?,?,?,?,?)""",
                   (file_uid, str(path), size, mtime, content_sig, VERSION_KEY, time.time(), 'active'))

        if own_con:
            con.commit()

        if lss_config.DEBUG:
            print(f"Indexed: {len(fresh_spans)} new spans, {len(stale_hashes)} removed")

        _cache_put(_get_file_cache_key(path), file_uid)
        return file_uid

    finally:
        if own_con:
            con.close()

# ── should_exclude (used by lss_sync.py) ─────────────────────────────────────

def _should_exclude_path(file_path, base_path):
    """Check if path should be excluded based on EXCLUDED_DIRS and user config."""
    try:
        rel_path = Path(file_path).relative_to(base_path)
        for part in rel_path.parts:
            if part in EXCLUDED_DIRS:
                return True
    except ValueError:
        for part in Path(file_path).parts:
            if part in EXCLUDED_DIRS:
                return True

    if Path(file_path).name in EXCLUDED_FILES:
        return True

    cfg = lss_config.load_config()
    for pattern in cfg.get("exclude_patterns", []):
        name = Path(file_path).name
        try:
            rel = str(Path(file_path).relative_to(base_path))
        except ValueError:
            rel = str(file_path)
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
            return True
        if '/' not in pattern and '*' not in pattern and '?' not in pattern:
            try:
                parts = Path(file_path).relative_to(base_path).parts
            except ValueError:
                parts = Path(file_path).parts
            if pattern in parts:
                return True

    return False

# ── Discovery + batch indexing ───────────────────────────────────────────────

def discover_files(paths_or_dir):
    """Discover indexable text files without indexing them.

    Uses os.walk with directory pruning (never enters node_modules etc).
    Returns (all_paths, new_paths, already_indexed_count).
    """
    if isinstance(paths_or_dir, (str, Path)):
        path = Path(paths_or_dir)
        if path.is_dir():
            all_paths = list(_walk_text_files(path))
        else:
            all_paths = [path]
    else:
        all_paths = [Path(p) for p in paths_or_dir]

    # Batch check: single DB connection for all files
    new_paths = []
    already = 0
    con = _init_db()
    try:
        for p in all_paths:
            resolved = p.resolve()
            try:
                cache_key = _get_file_cache_key(resolved)
            except OSError:
                new_paths.append(p)
                continue
            if cache_key in _file_cache:
                _file_cache.move_to_end(cache_key)
                already += 1
                continue
            try:
                stat = resolved.stat()
                row = con.execute(
                    "SELECT file_uid FROM files WHERE path = ? AND size = ? AND mtime = ? AND version = ?",
                    (str(resolved), stat.st_size, stat.st_mtime, VERSION_KEY)
                ).fetchone()
                if row:
                    _cache_put(cache_key, row[0])
                    already += 1
                else:
                    new_paths.append(p)
            except OSError:
                new_paths.append(p)
    finally:
        con.close()

    return all_paths, new_paths, already


def ingest_many(paths_or_dir, progress_cb=None):
    """Ingest multiple files or a directory.

    When given a list of paths (e.g. from discover_files), indexes them directly —
    no re-walking, no re-filtering.  Uses a SINGLE DB connection and commits in
    batches for speed.

    Args:
        paths_or_dir: A directory path, file path, or list of paths.
        progress_cb:  Optional callback(current, total, path) called after each file.
    """
    if isinstance(paths_or_dir, (str, Path)):
        path = Path(paths_or_dir)
        if path.is_dir():
            paths = list(_walk_text_files(path))
        else:
            paths = [path]
    else:
        paths = [Path(p) for p in paths_or_dir]

    total = len(paths)
    if total == 0:
        return []

    # Single connection for the entire batch
    con = _init_db()
    file_uids = []
    batch_count = 0
    COMMIT_INTERVAL = 50  # commit every N files for WAL performance

    try:
        for i, fpath in enumerate(paths):
            try:
                fpath_resolved = fpath.resolve()
                # Fast-path: check cache/DB before doing expensive work
                try:
                    cache_key = _get_file_cache_key(fpath_resolved)
                    if cache_key in _file_cache:
                        _file_cache.move_to_end(cache_key)
                        file_uids.append(_file_cache[cache_key])
                        if progress_cb:
                            progress_cb(i + 1, total, fpath)
                        continue
                except OSError:
                    pass

                file_uid = _do_index(fpath_resolved, con=con)
                file_uids.append(file_uid)
                batch_count += 1

                # Periodic commit to keep WAL size manageable
                if batch_count >= COMMIT_INTERVAL:
                    con.commit()
                    batch_count = 0

            except Exception as e:
                if lss_config.DEBUG:
                    print(f"Failed to index {fpath}: {e}")

            if progress_cb:
                progress_cb(i + 1, total, fpath)

        # Final commit
        if batch_count > 0:
            con.commit()

        # Housekeeping — best-effort
        try:
            con.execute("INSERT INTO fts(fts) VALUES('optimize')")
            con.commit()
        except sqlite3.OperationalError:
            pass
        try:
            con.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.OperationalError:
            pass

    finally:
        con.close()

    return file_uids

# ── Maintenance / query API ──────────────────────────────────────────────────

def sweep(retention_days=30):
    """Clean up missing files and orphaned data; prune embeddings by age."""
    con = _init_db()
    cur = con.cursor()
    try:
        files = cur.execute("SELECT file_uid, path FROM files WHERE status = 'active'").fetchall()
        missing_uids = [file_uid for file_uid, path in files if not Path(path).exists()]

        if missing_uids:
            placeholders = ",".join(["?"] * len(missing_uids))
            cur.execute(f"UPDATE files SET status = 'missing' WHERE file_uid IN ({placeholders})", missing_uids)
            cur.execute(f"DELETE FROM fts WHERE file_uid IN ({placeholders})", missing_uids)

        cutoff_time = time.time() - (retention_days * 24 * 3600)
        cur.execute(
            """DELETE FROM embeddings
               WHERE text_hash NOT IN (SELECT DISTINCT text_hash FROM fts)
                  OR created < ?""",
            (cutoff_time,),
        )

        con.commit()
        if lss_config.DEBUG:
            print(f"Swept: {len(missing_uids)} missing files, orphaned embeddings cleaned")
    finally:
        con.close()

def get_db_path():
    """Get the LSS database path"""
    return str(LSS_DB)

def prep_file(file_path):
    """Lightweight prep: ensure file is indexed (for batch scenarios)"""
    return ensure_indexed(file_path, force_reindex=False)

def get_file_uid(file_path):
    """Fast lookup of file_uid without indexing (read-only)"""
    path = Path(file_path).resolve()

    try:
        cache_key = _get_file_cache_key(path)
        if cache_key in _file_cache:
            _file_cache.move_to_end(cache_key)
            return _file_cache[cache_key]
    except:
        pass

    con = _init_db()
    try:
        stat = path.stat()
        row = con.execute(
            "SELECT file_uid FROM files WHERE path = ? AND size = ? AND mtime = ? AND version = ?",
            (str(path), stat.st_size, stat.st_mtime, VERSION_KEY)
        ).fetchone()
        if row:
            return row[0]
    except:
        pass
    finally:
        con.close()

    return None

def remove_files(paths):
    """Remove one or more files from the index."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [str(Path(p).resolve()) for p in paths]

    con = _init_db()
    cur = con.cursor()
    removed = 0
    try:
        for p in paths:
            rows = cur.execute("SELECT file_uid FROM files WHERE path = ?", (p,)).fetchall()
            for (file_uid,) in rows:
                hashes = [r[0] for r in cur.execute("SELECT text_hash FROM fts WHERE file_uid = ?", (file_uid,)).fetchall()]
                if hashes:
                    placeholders = ",".join("?" * len(hashes))
                    cur.execute(f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})", hashes)
                cur.execute("DELETE FROM fts WHERE file_uid = ?", (file_uid,))
                cur.execute("DELETE FROM files WHERE file_uid = ?", (file_uid,))
                removed += 1
        con.commit()
    finally:
        con.close()
        _file_cache.clear()
    return removed

def clear_embeddings(days=None):
    """Delete embeddings entirely or older than N days."""
    con = _init_db()
    try:
        if days is None:
            con.execute("DELETE FROM embeddings")
        else:
            cutoff = time.time() - days * 24 * 3600
            con.execute("DELETE FROM embeddings WHERE created < ?", (cutoff,))
        con.commit()
    finally:
        con.close()

def clear_all():
    """Drop the entire LSS database (and WAL/SHM), then recreate empty schema."""
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(LSS_DB) + suffix).unlink(missing_ok=True)
        except Exception:
            pass
    _file_cache.clear()
    con = _init_db()
    con.close()
