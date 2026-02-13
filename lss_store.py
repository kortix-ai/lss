import os, time, sqlite3, hashlib, json, re, fnmatch
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from lss_config import LSS_DIR, LSS_DB, VERSION_KEY
import lss_config
import lss_extract

# ── File filtering ────────────────────────────────────────────────────────────
#
# INCLUSION-BASED: Only index files with known text/code/doc extensions.
# Unknown extensions are SKIPPED by default (no byte-level guessing).
# Users can add more via `lss include add <ext>`.
# Binary extensions are a fast-reject layer (never even stat the file).
# Excluded directories are pruned during os.walk (never entered).

# Extensions we KNOW contain indexable text or have dedicated extractors.
INDEXED_EXTENSIONS = {
    # Text / documentation
    '.txt', '.md', '.markdown', '.rst', '.org', '.adoc', '.tex', '.rtf',
    '.text', '.log',
    # Code — Python
    '.py', '.pyi', '.pyw',
    # Code — JavaScript / TypeScript
    '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs', '.vue', '.svelte',
    # Code — Systems
    '.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cxx', '.hxx',
    '.go', '.rs', '.zig',
    # Code — JVM
    '.java', '.kt', '.kts', '.scala', '.clj', '.cljs', '.groovy', '.gradle',
    # Code — .NET
    '.cs', '.fs', '.vb', '.csproj', '.fsproj',
    # Code — Scripting
    '.rb', '.php', '.pl', '.pm', '.lua', '.r', '.R', '.jl',
    # Code — Mobile / Apple
    '.swift', '.m', '.mm',
    # Code — Functional
    '.hs', '.ml', '.mli', '.ex', '.exs', '.erl',
    # Shell / scripts
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
    # Web
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.xml', '.xsl', '.xslt', '.svg',
    # Data (text-based)
    '.json', '.jsonl', '.ndjson',
    '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.properties',
    '.csv', '.tsv',
    '.env.example', '.env.template',
    # Query / schema
    '.sql', '.graphql', '.gql', '.prisma',
    # Config files
    '.editorconfig', '.eslintrc', '.prettierrc', '.babelrc',
    '.npmrc', '.yarnrc',
    # Documents with dedicated extractors (lss_extract.py)
    '.pdf', '.docx', '.xlsx', '.pptx', '.eml',
    # Misc text
    '.diff', '.patch',
}

# Well-known filenames WITHOUT extensions that we should index.
KNOWN_EXTENSIONLESS = {
    'Makefile', 'GNUmakefile', 'makefile',
    'Dockerfile', 'Containerfile',
    'Vagrantfile', 'Procfile', 'Brewfile',
    'Gemfile', 'Rakefile', 'Guardfile',
    'LICENSE', 'LICENCE', 'COPYING', 'NOTICE',
    'README', 'CHANGELOG', 'CHANGES', 'HISTORY', 'AUTHORS', 'CONTRIBUTORS',
    'INSTALL', 'TODO', 'HACKING',
    '.gitignore', '.gitattributes', '.gitmodules',
    '.dockerignore', '.editorconfig', '.eslintignore',
    '.prettierignore', '.npmignore', '.slugignore',
    '.flake8', '.pylintrc', '.rubocop.yml',
    '.clang-format', '.clang-tidy',
    'CMakeLists.txt',  # has extension but worth calling out
    'go.mod', 'go.sum',
    'requirements.txt', 'constraints.txt',
    'setup.py', 'setup.cfg',
}

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
    # Documents (binary formats we CAN'T extract — formats we CAN extract
    # are handled by lss_extract and NOT listed here)
    '.doc',  # legacy Word (needs LibreOffice — too heavy)
    '.ppt',  # legacy PowerPoint
    '.xls',  # legacy Excel (openpyxl handles .xlsx only)
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


def _path_uid(resolved_path):
    """Generate a unique file_uid from the resolved absolute path.

    Uses a hash of the path so that different files with the same content
    get distinct file_uid values (the old content-based uid caused
    INSERT OR REPLACE collisions — only one path survived per content hash).
    """
    return "f_" + hashlib.md5(str(resolved_path).encode('utf-8')).hexdigest()

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
    # Binary-format files with dedicated extractors
    if ext in ('.pdf', '.docx', '.xlsx', '.pptx', '.eml'):
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

def _extract_text(file_path):
    """Extract text from supported file types.

    Delegates to lss_extract.extract_text() which handles PDF, DOCX, XLSX,
    PPTX, HTML, EML, JSON, CSV, and plain text with format-specific parsers.
    """
    return lss_extract.extract_text(str(file_path))

def _normalize_text(text):
    """Normalize text for indexing (FTS storage).

    Chunking should run on text that still has newlines so markdown/python
    boundary detection works. This function is for the stored chunk text.
    """
    import unicodedata
    text = unicodedata.normalize('NFKC', text or "")
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_for_chunking(text: str) -> str:
    """Normalize text for chunk boundary detection.

    Keeps newlines intact (no whitespace collapsing) so _chunk_markdown and
    _chunk_python can detect headings/definitions.
    """
    import unicodedata
    t = unicodedata.normalize('NFKC', text or "")
    return t.replace("\r\n", "\n").replace("\r", "\n")

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


def _chunk_markdown(text, max_words=220, stride=200):
    """Chunk markdown/rst by heading boundaries.

    Splits on lines starting with # (markdown) or underline patterns (rst).
    Each chunk includes the heading + its content.
    Falls back to _span_chunk if no headings are found.
    """
    import re as _re

    # Split on markdown headings (# ... or ## ... etc.)
    # Also handles RST underline headings (===, ---, ~~~)
    heading_pattern = _re.compile(
        r'^(#{1,6}\s+.+)$|^(.+)\n([=\-~]{3,})$',
        _re.MULTILINE
    )

    # Find all heading positions
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return _span_chunk(text, max_words, stride)

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    # Include any preamble before the first heading
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.insert(0, preamble)

    # If sections are too long, sub-chunk them with word-window
    chunks = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            chunks.append(("markdown", section))
        else:
            # Sub-chunk long sections
            for chunk_type, chunk_text in _span_chunk(section, max_words, stride):
                chunks.append(("markdown", chunk_text))

    return chunks if chunks else _span_chunk(text, max_words, stride)


def _chunk_python(text, max_words=220, stride=200):
    """Chunk Python code by function/class boundaries.

    Uses regex-based splitting (not ast) to handle incomplete/invalid code.
    Each chunk includes the def/class line + its body.
    Falls back to _span_chunk if no definitions are found.
    """
    import re as _re

    # Split on top-level def/class (lines starting with def or class, no indent)
    # Also match methods (indented def) for large classes
    defn_pattern = _re.compile(r'^((?:class|def)\s+\w+)', _re.MULTILINE)
    matches = list(defn_pattern.finditer(text))

    if not matches:
        return _span_chunk(text, max_words, stride)

    sections = []

    # Include preamble (imports, constants) before first definition
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(preamble)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    # Sub-chunk long sections
    chunks = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            chunks.append(("python", section))
        else:
            for chunk_type, chunk_text in _span_chunk(section, max_words, stride):
                chunks.append(("python", chunk_text))

    return chunks if chunks else _span_chunk(text, max_words, stride)


def _smart_chunk(text, file_ext, words_per_span=220, stride=200):
    """Dispatch to format-specific chunkers based on file extension.

    Args:
        text: The extracted text to chunk.
        file_ext: File extension (e.g., ".md", ".py", ".txt").
        words_per_span: Max words per chunk (for word-window fallback).
        stride: Word stride between chunks (for word-window fallback).

    Returns:
        List of (chunk_type, chunk_text) tuples.
    """
    if not text or not text.strip():
        return []

    ext = file_ext.lower()

    # Markdown / RST — split on headings
    if ext in ('.md', '.markdown', '.rst', '.adoc', '.org'):
        return _chunk_markdown(text, words_per_span, stride)

    # Python — split on def/class boundaries
    if ext in ('.py', '.pyi', '.pyw'):
        return _chunk_python(text, words_per_span, stride)

    # Everything else — standard word-window chunking
    return _span_chunk(text, words_per_span, stride)

# ── .gitignore parsing ───────────────────────────────────────────────────────

def _parse_gitignore(gitignore_path):
    """Parse a .gitignore file into (file_patterns, dir_patterns) lists."""
    file_patterns = []
    dir_patterns = []
    try:
        text = Path(gitignore_path).read_text(encoding='utf-8', errors='replace')
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Negation patterns (!) are too complex — skip them
            if line.startswith('!'):
                continue
            if line.endswith('/'):
                # Directory pattern
                dir_patterns.append(line.rstrip('/'))
            else:
                file_patterns.append(line)
    except Exception:
        pass
    return file_patterns, dir_patterns

# ── File walking (os.walk with inclusion-based filtering) ────────────────────

def _walk_text_files(base_path, extra_exclude_patterns=None):
    """Walk a directory tree with INCLUSION-BASED filtering.

    Strategy:
    1. Prune excluded directories in-place (os.walk never enters them)
    2. Parse .gitignore files in each directory for additional exclusions
    3. Fast-reject files by BINARY_EXTENSIONS or EXCLUDED_FILES
    4. Accept files with known INDEXED_EXTENSIONS or KNOWN_EXTENSIONLESS names
    5. For unknown extensions: skip (no byte-level guessing)
    6. Extensionless files: accept if name is in KNOWN_EXTENSIONLESS, else byte-check

    Users can extend the allowlist via `lss include add <ext>`.
    """
    base = str(Path(base_path).resolve())

    # Pre-compute user config (load ONCE, not per-file)
    cfg = lss_config.load_config()
    cfg_patterns = list(extra_exclude_patterns or [])
    cfg_patterns.extend(cfg.get("exclude_patterns", []))

    # User-added include extensions
    user_include = set(cfg.get("include_extensions", []))

    # Build combined set: default + user includes
    include_exts = INDEXED_EXTENSIONS | user_include

    # Split exclude patterns into dir-name patterns and file-glob patterns
    dir_excludes = set(EXCLUDED_DIRS)
    file_globs = []
    for pat in cfg_patterns:
        if '/' not in pat and '*' not in pat and '?' not in pat:
            dir_excludes.add(pat)
        file_globs.append(pat)

    # Gitignore state as a true stack: [(dir_path, file_patterns, dir_patterns)]
    # Keep bounded by directory depth for performance.
    gitignore_stack = []

    def _is_subpath(path: str, parent: str) -> bool:
        if path == parent:
            return True
        parent_prefix = parent.rstrip(os.sep) + os.sep
        return path.startswith(parent_prefix)

    for root, dirs, files in os.walk(base):
        # Maintain stack as we descend/ascend
        while gitignore_stack and not _is_subpath(root, gitignore_stack[-1][0]):
            gitignore_stack.pop()

        # ── Parse .gitignore in this directory ──
        gitignore_file = os.path.join(root, '.gitignore')
        if os.path.isfile(gitignore_file):
            gi_files, gi_dirs = _parse_gitignore(gitignore_file)
            gitignore_stack.append((root, gi_files, gi_dirs))

        # ── Prune excluded directories ──
        # Combine hardcoded excludes + active gitignore dir patterns
        gi_dir_excludes = set()
        if gitignore_stack:
            for _, _, gi_dirs_pats in gitignore_stack:
                gi_dir_excludes.update(gi_dirs_pats)

        dirs[:] = [
            d for d in dirs
            if d not in dir_excludes and d not in gi_dir_excludes
        ]

        # ── Collect active gitignore file patterns for this directory ──
        active_gi_file_patterns = []
        if gitignore_stack:
            for _, gi_file_pats, _ in gitignore_stack:
                active_gi_file_patterns.extend(gi_file_pats)

        for name in files:
            # 1. Fast-reject: excluded filenames
            if name in EXCLUDED_FILES:
                continue

            ext = os.path.splitext(name)[1].lower()

            # 2. Fast-reject: binary extensions
            if ext in BINARY_EXTENSIONS:
                continue

            # 3. Check user-configured file glob exclusion patterns
            if file_globs:
                skip = False
                for pat in file_globs:
                    if fnmatch.fnmatch(name, pat):
                        skip = True
                        break
                if skip:
                    continue

            # 4. Check gitignore file patterns
            if active_gi_file_patterns:
                skip = False
                for pat in active_gi_file_patterns:
                    if fnmatch.fnmatch(name, pat):
                        skip = True
                        break
                if skip:
                    continue

            # 5. Inclusion check: known extension OR known extensionless name
            if ext:
                # Has an extension — check against allowlist
                if ext not in include_exts:
                    continue  # Unknown extension → skip
            else:
                # No extension — check against known extensionless names
                if name not in KNOWN_EXTENSIONLESS:
                    # Unknown extensionless file — byte-check as fallback
                    full = os.path.join(root, name)
                    try:
                        with open(full, 'rb') as f:
                            chunk = f.read(8192)
                        if not chunk or b'\0' in chunk:
                            continue
                        chunk.decode('utf-8')
                    except Exception:
                        continue
                    # Passed byte-check, yield below
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue
                    if size == 0 or size > MAX_FILE_SIZE:
                        continue
                    yield Path(full)
                    continue

            full = os.path.join(root, name)

            # 6. Size check (no open, just stat)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size == 0 or size > MAX_FILE_SIZE:
                continue

            yield Path(full)

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
        file_uid = _path_uid(path)

        row = cur.execute(
            "SELECT content_sig, version FROM files WHERE file_uid = ?",
            (file_uid,)
        ).fetchone()

        if row and row[0] == content_sig and row[1] == VERSION_KEY:
            # Content unchanged — update mtime/size in case the inode was
            # touched (e.g. git checkout, cp --preserve) so that the fast
            # path+size+mtime lookup in discover_files keeps matching.
            cur.execute(
                "UPDATE files SET path = ?, size = ?, mtime = ? WHERE file_uid = ?",
                (str(path), size, mtime, file_uid),
            )
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
        chunk_input = _normalize_for_chunking(raw_text)
        spans = _smart_chunk(chunk_input, path.suffix)

        existing_hashes = set()
        if row:
            existing_rows = cur.execute(
                "SELECT text_hash FROM fts WHERE file_uid = ?", (file_uid,)
            ).fetchall()
            existing_hashes = {r[0] for r in existing_rows}

        new_spans = []
        new_hashes = set()
        for i, (chunk_type, span_text) in enumerate(spans):
            span_text = _normalize_text(span_text)
            span_id = f"{file_uid}::{i}" if len(spans) > 1 else file_uid
            text_hash_val = _text_hash(span_text)
            new_spans.append((span_id, span_text, file_uid, str(path), text_hash_val))
            new_hashes.add(text_hash_val)

        stale_hashes = existing_hashes - new_hashes
        fresh_hashes = new_hashes - existing_hashes

        if stale_hashes:
            placeholders = ','.join(['?' for _ in stale_hashes])
            cur.execute(
                f"DELETE FROM fts WHERE file_uid = ? AND text_hash IN ({placeholders})",
                [file_uid] + list(stale_hashes),
            )
            cur.execute(
                f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})",
                list(stale_hashes),
            )

        fresh_spans = [s for s in new_spans if s[4] in fresh_hashes]
        if fresh_spans:
            cur.executemany(
                "INSERT INTO fts(id, text, file_uid, file_path, text_hash) VALUES(?,?,?,?,?)",
                fresh_spans,
            )

        cur.execute(
            """INSERT OR REPLACE INTO files
                      (file_uid, path, size, mtime, content_sig, version, indexed_at, status)
                      VALUES (?,?,?,?,?,?,?,?)""",
            (file_uid, str(path), size, mtime, content_sig, VERSION_KEY, time.time(), 'active'),
        )

        if own_con:
            con.commit()

        if lss_config.DEBUG:
            print(f"Indexed: {len(fresh_spans)} new spans, {len(stale_hashes)} removed")

        _cache_put(_get_file_cache_key(path), file_uid)
        return file_uid

    finally:
        if own_con:
            con.close()


def _default_index_jobs(total_files: int) -> int:
    """Worker thread count for preprocessing.

    Auto-enables for large batches; DB writes remain single-threaded.
    Override with env var LSS_INDEX_JOBS (0 disables).
    """
    env = os.environ.get("LSS_INDEX_JOBS")
    if env is not None:
        try:
            v = int(env)
            return max(0, v)
        except ValueError:
            return 0

    if total_files < 500:
        return 0

    cpu = os.cpu_count() or 4
    return min(16, max(4, cpu))


def _commit_interval(total_files: int) -> int:
    env = os.environ.get("LSS_COMMIT_INTERVAL")
    if env is not None:
        try:
            v = int(env)
            return max(1, v)
        except ValueError:
            pass

    if total_files >= 2000:
        return 500
    if total_files >= 500:
        return 250
    return 50


_BINARY_EXTRACTOR_EXTS = {'.pdf', '.docx', '.xlsx', '.pptx'}


def _extract_text_from_bytes(path: Path, data: bytes) -> str:
    """Extract text using already-read bytes when possible.

    For binary formats with dedicated parsers, fall back to the existing
    file-path based extractors.
    """
    ext = path.suffix.lower()
    if ext in _BINARY_EXTRACTOR_EXTS:
        return _extract_text(path)

    # Decode once for all text formats.
    def _decode(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            for enc in ("latin-1", "cp1252", "iso-8859-1"):
                try:
                    return b.decode(enc)
                except UnicodeDecodeError:
                    continue
        return ""

    text = _decode(data)
    if not text:
        return ""

    try:
        if ext == ".json":
            obj = json.loads(text)
            if isinstance(obj, dict):
                return " ".join(str(v) for v in obj.values() if isinstance(v, (str, int, float)))
            if isinstance(obj, list):
                return " ".join(str(item) for item in obj if isinstance(item, (str, int, float)))
            return str(obj)

        if ext in (".jsonl", ".ndjson"):
            out = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    if "text" in obj and isinstance(obj.get("text"), str):
                        out.append(obj["text"])
                    else:
                        out.append(" ".join(str(v) for v in obj.values() if isinstance(v, (str, int, float))))
                elif isinstance(obj, (str, int, float)):
                    out.append(str(obj))
            return "\n".join(out)

        if ext in (".csv", ".tsv"):
            import csv as _csv
            import io as _io
            delim = "\t" if ext == ".tsv" else ","
            reader = _csv.reader(_io.StringIO(text), delimiter=delim)
            rows = list(reader)
            if not rows:
                return ""
            result = " ".join(rows[0])
            for row in rows[1: min(100, len(rows))]:
                result += " " + " ".join(row)
            return result

        if ext in (".html", ".htm"):
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                clean = soup.get_text(separator="\n", strip=True)
                clean = re.sub(r"\n{3,}", "\n\n", clean)
                return clean.strip()
            except Exception:
                return re.sub(r"<[^>]+>", " ", text)

        if ext == ".eml":
            try:
                import email as email_mod
                from email.policy import default as default_policy
                msg = email_mod.message_from_bytes(data, policy=default_policy)
                parts = []
                subject = msg.get("Subject", "")
                if subject:
                    parts.append(f"Subject: {subject}")
                body_text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_content()
                            if isinstance(payload, str):
                                body_text = payload
                                break
                else:
                    payload = msg.get_content()
                    if isinstance(payload, str):
                        body_text = payload
                if body_text:
                    parts.append(body_text.strip())
                return "\n\n".join(parts)
            except Exception:
                return text

        return text
    except Exception:
        return ""


def _apply_prepared_index(prep: dict, con: sqlite3.Connection):
    """Write a prepared file payload into SQLite (single-writer)."""
    path = prep["path"]
    file_uid = prep["file_uid"]
    size = prep["size"]
    mtime = prep["mtime"]
    content_sig = prep["content_sig"]
    spans = prep["spans"]

    cur = con.cursor()

    row = cur.execute(
        "SELECT content_sig, version FROM files WHERE file_uid = ?",
        (file_uid,),
    ).fetchone()

    if row and row[0] == content_sig and row[1] == VERSION_KEY:
        cur.execute(
            "UPDATE files SET path = ?, size = ?, mtime = ? WHERE file_uid = ?",
            (str(path), size, mtime, file_uid),
        )
        _cache_put(_get_file_cache_key(path), file_uid)
        return file_uid

    # Clean up old entries for this path (handles content changes)
    old_entries = cur.execute(
        "SELECT file_uid FROM files WHERE path = ? AND file_uid != ?",
        (str(path), file_uid),
    ).fetchall()
    for (old_uid,) in old_entries:
        old_hashes = cur.execute(
            "SELECT text_hash FROM fts WHERE file_uid = ?", (old_uid,)
        ).fetchall()
        cur.execute("DELETE FROM fts WHERE file_uid = ?", (old_uid,))
        if old_hashes:
            placeholders = ','.join(['?' for _ in old_hashes])
            cur.execute(
                f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})",
                [h[0] for h in old_hashes],
            )
        cur.execute("DELETE FROM files WHERE file_uid = ?", (old_uid,))

    existing_hashes = set()
    if row:
        existing_rows = cur.execute(
            "SELECT text_hash FROM fts WHERE file_uid = ?", (file_uid,)
        ).fetchall()
        existing_hashes = {r[0] for r in existing_rows}

    new_spans = []
    new_hashes = set()
    for i, (_chunk_type, span_text) in enumerate(spans):
        if not span_text:
            continue
        span_id = f"{file_uid}::{i}" if len(spans) > 1 else file_uid
        text_hash_val = _text_hash(span_text)
        new_spans.append((span_id, span_text, file_uid, str(path), text_hash_val))
        new_hashes.add(text_hash_val)

    stale_hashes = existing_hashes - new_hashes
    fresh_hashes = new_hashes - existing_hashes

    if stale_hashes:
        placeholders = ','.join(['?' for _ in stale_hashes])
        cur.execute(
            f"DELETE FROM fts WHERE file_uid = ? AND text_hash IN ({placeholders})",
            [file_uid] + list(stale_hashes),
        )
        cur.execute(
            f"DELETE FROM embeddings WHERE text_hash IN ({placeholders})",
            list(stale_hashes),
        )

    fresh_spans = [s for s in new_spans if s[4] in fresh_hashes]
    if fresh_spans:
        cur.executemany(
            "INSERT INTO fts(id, text, file_uid, file_path, text_hash) VALUES(?,?,?,?,?)",
            fresh_spans,
        )

    cur.execute(
        """INSERT OR REPLACE INTO files
                      (file_uid, path, size, mtime, content_sig, version, indexed_at, status)
                      VALUES (?,?,?,?,?,?,?,?)""",
        (file_uid, str(path), size, mtime, content_sig, VERSION_KEY, time.time(), 'active'),
    )

    _cache_put(_get_file_cache_key(path), file_uid)
    return file_uid

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

    # Batch check with a single DB connection.
    # For directory inputs, fetch indexed rows via a single prefix query to
    # avoid one SELECT per file.
    new_paths = []
    already = 0
    con = _init_db()
    try:
        indexed_map = None
        if isinstance(paths_or_dir, (str, Path)):
            base_path = Path(paths_or_dir)
            if base_path.is_dir():
                base_abs = str(base_path.resolve())
                prefix = base_abs.rstrip(os.sep) + os.sep
                rows = con.execute(
                    """SELECT path, size, mtime, file_uid FROM files
                       WHERE status='active' AND version=? AND (path = ? OR path LIKE ?)""",
                    (VERSION_KEY, base_abs, prefix + "%"),
                ).fetchall()
                indexed_map = {r[0]: (r[1], r[2], r[3]) for r in rows}

        for p in all_paths:
            resolved = p.resolve()
            try:
                stat = resolved.stat()
                cache_key = (str(resolved), stat.st_size, int(stat.st_mtime), VERSION_KEY)
            except OSError:
                new_paths.append(p)
                continue

            if cache_key in _file_cache:
                _file_cache.move_to_end(cache_key)
                already += 1
                continue

            if indexed_map is not None:
                hit = indexed_map.get(str(resolved))
                if hit and hit[0] == stat.st_size and hit[1] == stat.st_mtime:
                    _cache_put(cache_key, hit[2])
                    already += 1
                else:
                    new_paths.append(p)
                continue

            # Fallback for non-directory inputs
            row = con.execute(
                "SELECT file_uid FROM files WHERE path = ? AND size = ? AND mtime = ? AND version = ?",
                (str(resolved), stat.st_size, stat.st_mtime, VERSION_KEY)
            ).fetchone()
            if row:
                _cache_put(cache_key, row[0])
                already += 1
            else:
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

    con = _init_db()
    file_uids = []
    batch_count = 0
    COMMIT_INTERVAL = _commit_interval(total)
    jobs = _default_index_jobs(total)

    def _prepare(resolved_path: Path) -> dict:
        stat = resolved_path.stat()
        size = stat.st_size
        if size == 0 or size > MAX_FILE_SIZE:
            raise ValueError("skip")
        data = resolved_path.read_bytes()
        content_sig = hashlib.md5(data).hexdigest()
        raw_text = _extract_text_from_bytes(resolved_path, data)
        chunk_input = _normalize_for_chunking(raw_text)
        spans = _smart_chunk(chunk_input, resolved_path.suffix)
        spans = [(ct, _normalize_text(st)) for ct, st in spans]
        return {
            "path": resolved_path,
            "size": size,
            "mtime": stat.st_mtime,
            "file_uid": _path_uid(resolved_path),
            "content_sig": content_sig,
            "spans": spans,
        }

    try:
        con.execute("BEGIN")

        processed = 0
        pending = []

        # Main-thread cache fast-path (keeps LRU consistent and avoids wasted work)
        for fpath in paths:
            try:
                resolved = fpath.resolve()
                st = resolved.stat()
                cache_key = (str(resolved), st.st_size, int(st.st_mtime), VERSION_KEY)
            except OSError:
                pending.append((fpath, None))
                continue

            if cache_key in _file_cache:
                _file_cache.move_to_end(cache_key)
                file_uids.append(_file_cache[cache_key])
                processed += 1
                if progress_cb:
                    progress_cb(processed, total, fpath)
                continue

            pending.append((fpath, resolved))

        if jobs > 0 and pending:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                fut_map = {}
                for original, resolved in pending:
                    if resolved is None:
                        # stat/resolve failed earlier
                        processed += 1
                        if progress_cb:
                            progress_cb(processed, total, original)
                        continue
                    fut = ex.submit(_prepare, resolved)
                    fut_map[fut] = original

                for fut in as_completed(fut_map):
                    original = fut_map[fut]
                    try:
                        prep = fut.result()
                        uid = _apply_prepared_index(prep, con)
                        if uid:
                            file_uids.append(uid)
                            batch_count += 1

                        if batch_count >= COMMIT_INTERVAL:
                            con.commit()
                            con.execute("BEGIN")
                            batch_count = 0
                    except Exception as e:
                        if lss_config.DEBUG and str(e) != "skip":
                            print(f"Failed to index {original}: {e}")
                    finally:
                        processed += 1
                        if progress_cb:
                            progress_cb(processed, total, original)
        else:
            # Sequential fallback (original behavior)
            for i, fpath in enumerate(paths):
                try:
                    fpath_resolved = fpath.resolve()
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

                    if batch_count >= COMMIT_INTERVAL:
                        con.commit()
                        con.execute("BEGIN")
                        batch_count = 0
                except Exception as e:
                    if lss_config.DEBUG:
                        print(f"Failed to index {fpath}: {e}")

                if progress_cb:
                    progress_cb(i + 1, total, fpath)

        con.commit()

        # Housekeeping — best-effort
        if total >= 200:
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
