# LSS Architecture & Pipeline Reference

> Local Semantic Search v0.5.0 — how it works, step by step, with real timing data.

---

## Overview

LSS finds content in your files by combining two complementary search strategies:

1. **BM25** (keyword match via SQLite FTS5) — fast, exact, zero-cost
2. **Embeddings** (semantic similarity via OpenAI or local fastembed) — understands meaning

Results from both are merged using **Reciprocal Rank Fusion (RRF)**, then re-ranked with **MMR** (Maximal Marginal Relevance) for diversity.

```
user query
    |
    v
[INDEXING PIPELINE]          [SEARCH PIPELINE]
file discovery               tokenize + DF lookup
    |                            |
inclusion filter             keyword extraction
    |                            |
document extraction          FTS5 BM25 query
    |                            |
normalize                    PRF expansion (optional)
    |                            |
smart chunking               embedding (OpenAI or local)
    |                            |
hash (MD5)                   embedding cache lookup
    |                            |
FTS5 insert + commit         RRF fusion
    |                            |
embedding cache write        MMR re-ranking
                                 |
                             output results
```

---

## Indexing Pipeline

Triggered by `lss "query" <dir>` (auto-index) or `lss index <dir>`.

### Step-by-step

| # | Step | What it does | Time (8 files, 59KB) | % |
|---|------|-------------|---------------------|---|
| 1 | **File discovery** | `os.walk` + in-place dir pruning + inclusion filter + .gitignore parsing | 16.0 ms | 25% |
| 2 | **Inclusion filter** | Check against `INDEXED_EXTENSIONS` allowlist (~80 exts) + `KNOWN_EXTENSIONLESS` + user includes | 0.1 ms | <1% |
| 3 | **Document extraction** | Dispatch by extension: plain text read, or PDF/DOCX/XLSX/PPTX/HTML/EML/JSON/CSV extraction | 0.7 ms | 1% |
| 4 | **Normalize** | Unicode NFKC + whitespace collapse | 1.1 ms | 2% |
| 5 | **Smart chunking** | Extension-aware: markdown heading splits, Python def/class splits, or word-window | 0.2 ms | <1% |
| 6 | **Hash** | MD5 content signature (file) + MD5 per chunk | 0.5 ms | 1% |
| 7 | **DB write** | FTS5 INSERT + files manifest + COMMIT | 45.6 ms | **71%** |
| | **TOTAL** | | **64.1 ms** | |
| | **Re-index (cached)** | Fast path: stat() + LRU cache check | 1.8 ms | |

### Key insight — Indexing

**DB writes dominate indexing (71%).** The actual text processing (read, extract, normalize, chunk, hash) takes <5 ms combined. SQLite FTS5 INSERT + WAL commit is the bottleneck. This is acceptable because:

- Indexing happens once per file (content-addressed via MD5)
- Re-indexing unchanged files hits the LRU fast path (~0.2 ms/file)
- `ingest_many` batches all files in a single transaction

### Re-index fast path

When a file hasn't changed (same path + size + mtime + version key):
1. Check in-memory LRU cache (O(1) hash lookup) → **0.01 ms**
2. If not in LRU, check SQLite `files` table with size/mtime guard → **0.2 ms**
3. No content hashing, no text extraction, no FTS writes

---

## Document Extraction Pipeline

New in v0.5.0. File: `lss_extract.py`.

The main dispatcher `extract_text(file_path)` routes by extension:

| Format | Library | What's extracted |
|--------|---------|-----------------|
| **PDF** | pdfminer.six | Layout-aware text from all pages |
| **DOCX** | python-docx | Paragraphs + table cells |
| **XLSX** | openpyxl | All sheets, row by row, cells tab-separated |
| **PPTX** | python-pptx | Slide text frames + table cells |
| **HTML** | beautifulsoup4 | Visible text (scripts/styles stripped) |
| **EML** | stdlib `email` | Subject + From + plain-text body |
| **JSON** | stdlib `json` | Pretty-printed (all values searchable) |
| **JSONL** | stdlib | Each line parsed + pretty-printed |
| **CSV** | stdlib `csv` | Rows formatted as "header: value" pairs |

All extraction functions return `""` on any error (never raise). If a library isn't installed, that format is silently skipped. Plain text files (code, markdown, config) are read directly via UTF-8.

---

## Smart Chunking

New in v0.5.0. Replaced the fixed word-window approach.

`_smart_chunk(text, ext)` dispatches by file extension:

### Markdown chunking (`.md`)

Splits on heading lines (`# ...`, `## ...`, etc.):
1. Walk lines, accumulate into sections
2. Each `# heading` starts a new chunk
3. Chunks smaller than threshold are merged with the next section
4. Chunks larger than the word-window limit are sub-chunked with word-window

### Python chunking (`.py`)

Splits on definition boundaries:
1. Walk lines, split on `def ` and `class ` at the start of a line
2. Each definition starts a new chunk (includes decorators if adjacent)
3. Top-of-file imports/comments form their own chunk
4. Oversized chunks are sub-chunked with word-window

### Default chunking (everything else)

Sliding word-window:
- **Span size:** 220 words (~1 paragraph of dense text)
- **Stride:** 200 words (20-word overlap between consecutive spans)
- **Why overlapping:** Ensures no sentence falls on a boundary and gets missed

A 1000-word file produces ~5 chunks. A 10,000-word file produces ~50 chunks.

### Per chunk

Each chunk gets:
- An MD5 hash (for deduplication and embedding cache lookup)
- An FTS5 entry (for BM25 keyword search)
- An embedding vector (computed lazily on first search, cached permanently)

---

## Inclusion-Based File Filtering

New in v0.5.0. Replaced the binary-detection approach with an **allowlist**.

### Why inclusion-based?

The old approach tried to detect binary files by reading bytes. This was slow, error-prone, and indexed junk (minified JS, generated code, data files with text-like content). The new approach only indexes files with **known extensions**.

### Filter layers

| Layer | What | Examples |
|-------|------|---------|
| 1. **Directory exclusions** | Entire trees pruned during `os.walk` | `node_modules/`, `.git/`, `__pycache__/`, `.venv/`, `dist/`, `build/` (~70 names) |
| 2. **`.gitignore` parsing** | Reads `.gitignore` in each subtree, applies patterns as additional excludes | `*.pyc`, `dist/`, `.env` |
| 3. **`INDEXED_EXTENSIONS`** | Allowlist of ~80 known text/code/doc extensions | `.py`, `.js`, `.md`, `.pdf`, `.docx`, `.yaml`, etc. |
| 4. **`KNOWN_EXTENSIONLESS`** | Named files without extensions | `Makefile`, `Dockerfile`, `LICENSE`, `README`, `.gitignore` |
| 5. **Excluded files** | Specific file names always skipped | `package-lock.json`, `yarn.lock`, `.DS_Store` |
| 6. **Max file size** | 2 MB default | Override with `LSS_MAX_FILE_SIZE` |
| 7. **User config** | `lss include add .ext` / `lss exclude add "*.log"` | Custom extensions and glob patterns |

**Unknown extensions are skipped by default.** This is the key design decision — it's better to miss a rare file type (user can add with `lss include add`) than to index thousands of junk files.

---

## Dual Embedding Provider Architecture

New in v0.5.0.

### Provider detection (`lss_config.py`)

`_detect_provider()` runs at import time:

1. Check `LSS_PROVIDER` env var → if set, use it
2. Check `~/.lss/config.json` `embedding_provider` field → if set, use it
3. Check `OPENAI_API_KEY` env var → if set, use `"openai"`
4. Check if `fastembed` is importable → if yes, use `"local"`
5. Fall back to `"openai"` (will fail with helpful error on first search)

### Module-level vars (`semantic_search.py`)

```python
EMBED_PROVIDER = lss_config.EMBEDDING_PROVIDER  # "openai" or "local"
EMBED_MODEL    = ...  # "text-embedding-3-small" or "BAAI/bge-small-en-v1.5"
EMBED_DIM      = ...  # 256 or 384
```

These replace the old `OPENAI_MODEL`/`OPENAI_DIM` constants. All cache keys use `EMBED_MODEL` and `EMBED_DIM`.

### The `_embed()` dispatcher

```python
def _embed(texts: list[str]) -> list[list[float]]:
    if EMBED_PROVIDER == "local":
        return _local_embed(texts)
    return _oai_embed(texts)
```

- `_oai_embed()`: OpenAI API call (batched, with retry/timeout)
- `_local_embed()`: fastembed `TextEmbedding` (lazy singleton, runs on CPU)

### VERSION_KEY

```python
VERSION_KEY = f"{EMBED_MODEL}:{EMBED_DIM}:p{PIPELINE_VERSION}:c{CHUNKING_VERSION}"
```

The version key incorporates the embedding provider. Switching providers (e.g., `lss config provider local`) changes the version key, which triggers re-embedding on next search. The BM25 index stays intact — only embedding vectors are recomputed.

### Lazy imports

OpenAI is imported inside `_oai_embed()`, not at module level. This means `import semantic_search` works even without `openai` installed (important for local-only users).

---

## Search Pipeline

Triggered by `lss "query"` or `lss search "query" -p <dir>`.

### Step-by-step

| # | Step | What it does | Cold (ms) | Warm (ms) | Hot (ms) |
|---|------|-------------|----------|----------|---------|
| 1 | **Tokenize + DF lookup** | Regex tokenize → batch `fts_vocab` query for document frequencies | 0.2 | 0.2 | 0.1 |
| 2 | **Keyword extraction** | DF-based filtering, phrase detection, short-query fast path | 0.1 | 0.1 | 0.1 |
| 3 | **FTS5 BM25 query** | `WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 600` | 0.4 | 0.4 | 0.3 |
| 4 | **PRF expansion** | Pseudo-relevance feedback: extract terms from top-10 docs, re-query | 0-2 | 0-2 | 0-1 |
| 5 | **Jaccard dedup** | Remove near-duplicate chunks (threshold=0.83) | <0.1 | <0.1 | <0.1 |
| 6 | **Embedding** | OpenAI API call or local fastembed inference | **565** / **50*** | **0.1**† | **0.1**† |
| 7 | **Embedding cache** | Write new vectors to SQLite `embeddings` table | 1-5 | 0 | 0 |
| 8 | **RRF fusion** | Reciprocal Rank Fusion of BM25 + embedding ranks | <0.1 | <0.1 | <0.1 |
| 9 | **Post-fusion boost** | Jaccard, phrase, digit co-mention features | <0.1 | <0.1 | <0.1 |
| 10 | **MMR re-ranking** | Vector-MMR (lambda=0.7) for diversity, if coverage >= 90% | <0.1 | <0.1 | <0.1 |
| | **TOTAL (OpenAI)** | | **~570** | **~157** | **~122** |
| | **TOTAL (local)** | | **~55** | **~5** | **~2** |

*\* Local cold: fastembed model load (~2s first time, then ~50ms inference)*
*† Warm/hot: embeddings served from SQLite cache or in-memory LRU*

### Key insight — Search

**With OpenAI:** The API call dominates cold search (90%+ of wall time). Everything else — BM25, fusion, MMR — takes <2 ms combined.

**With local embeddings:** No network dependency. Cold search is dominated by fastembed model load (one-time), then inference is ~50ms. Warm/hot searches are effectively instant.

### Caching strategy

| Cache layer | Scope | TTL | Lookup cost |
|-------------|-------|-----|-------------|
| **LRU (in-memory)** | Per-process, query + doc vectors | 15 min (query), 60 min (doc) | ~0.001 ms |
| **SQLite `embeddings` table** | Persistent, doc vectors only | Forever (until sweep) | ~0.1 ms |
| **OpenAI API** | N/A | N/A | **150-600 ms** |
| **fastembed inference** | N/A | N/A | **30-80 ms** |

### Three thermal states

- **Cold** (~570ms OpenAI / ~55ms local): First search ever, no caches. Hits API or runs inference for query + all doc embeddings.
- **Warm** (~157ms OpenAI / ~5ms local): Doc embeddings cached in SQLite. Only query embedding needed.
- **Hot** (~122ms OpenAI / ~2ms local): Everything in LRU. Zero API calls, zero inference. Pure local compute.

After a directory is searched once, subsequent searches are warm or hot.

---

## File Filtering — What Gets Indexed

LSS uses a multi-layer filter with an inclusion-based approach:

### Layer 1: Directory exclusions (`EXCLUDED_DIRS`)

Entire directory trees are pruned in-place during `os.walk`. This is the most impactful filter — it prevents walking into `node_modules/` (which can have 100K+ files).

```
.git, node_modules, __pycache__, .venv, venv, dist, build, target,
.next, .nuxt, .idea, .vscode, coverage, .cache, .terraform, .turbo,
.gradle, .mvn, bin, obj, packages, site-packages, logs, .lss, ...
```

Full list: ~70 directory names. See `EXCLUDED_DIRS` in `lss_store.py`.

### Layer 2: `.gitignore` parsing

During `os.walk`, lss reads `.gitignore` files in each subtree and applies their patterns as additional file/directory exclusions. This means generated files, build artifacts, and other gitignored content is automatically skipped.

### Layer 3: Inclusion-based extension filtering (`INDEXED_EXTENSIONS`)

~80 known text/code/document extensions are indexed. Unknown extensions are **skipped by default**.

**Code:** `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.lua`, `.r`, `.jl`, `.m`, `.sh`, `.bash`, `.zsh`, `.fish`, `.ps1`, `.bat`, `.cmd`, etc.

**Markup:** `.md`, `.rst`, `.tex`, `.html`, `.htm`, `.xml`, `.svg`, `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.cfg`, `.conf`

**Documents:** `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.eml`, `.csv`, `.tsv`, `.jsonl`

**Known extensionless:** `Makefile`, `Dockerfile`, `LICENSE`, `README`, `.gitignore`, `.dockerignore`, `.editorconfig`, etc.

### Layer 4: Excluded file names

Lock files, env files, OS junk:
```
package-lock.json, yarn.lock, pnpm-lock.yaml, poetry.lock,
Cargo.lock, go.sum, .DS_Store, .env, .env.local, ...
```

### Layer 5: Max file size

2 MB default. Files larger than this are skipped. Override with `LSS_MAX_FILE_SIZE=10485760` (10 MB).

### Layer 6: User-configured includes/excludes

```bash
lss include add .rst        # add a custom extension
lss include add .tex
lss exclude add "*.log"     # add a glob exclusion pattern
lss exclude add "generated"
```

---

## Database Schema

SQLite WAL mode, single file at `~/.lss/lss.db`.

```sql
-- File manifest (one row per indexed file)
CREATE TABLE files (
    file_uid    TEXT PRIMARY KEY,  -- f_{md5_of_content}
    path        TEXT NOT NULL,
    size        INTEGER,
    mtime       REAL,
    content_sig TEXT NOT NULL,     -- MD5 hex of file content
    version     TEXT NOT NULL,     -- "BAAI/bge-small-en-v1.5:384:p2:c4"
    indexed_at  REAL,
    status      TEXT DEFAULT 'active'  -- active | missing
);

-- Full-text search index (FTS5 with Porter stemmer)
CREATE VIRTUAL TABLE fts USING fts5(
    id UNINDEXED,        -- "{file_uid}::{chunk_index}" or just file_uid
    text,                -- chunk text (indexed, searchable)
    file_uid UNINDEXED,
    file_path UNINDEXED,
    text_hash UNINDEXED, -- MD5 binary, links to embeddings table
    tokenize='porter', prefix='2'
);

-- Vocabulary table for DF-based keyword filtering
CREATE VIRTUAL TABLE fts_vocab USING fts5vocab(fts, row);

-- Embedding vector cache (persistent across sessions)
CREATE TABLE embeddings (
    text_hash BLOB,       -- MD5 binary of chunk text
    model     TEXT,        -- "BAAI/bge-small-en-v1.5" or "text-embedding-3-small"
    dim       INTEGER,     -- 384 or 256
    version   TEXT,
    vector    BLOB,        -- float32 array
    created   REAL DEFAULT (unixepoch()),
    PRIMARY KEY (text_hash, model, dim, version)
);
```

### DB locking strategy

| Connection | Role | Auto-checkpoint | Busy timeout |
|------------|------|-----------------|--------------|
| `ingest_many` | Writer | `wal_autocheckpoint=4000` | 30s |
| `semantic_search` | Reader | `wal_autocheckpoint=0` (disabled) | 30s |
| `lss-sync` | Writer | `wal_autocheckpoint=4000` | 30s |
| Embedding cache write (from search) | Best-effort writer | N/A | try/except |

**Rule:** Writers use PASSIVE checkpoints only. Readers never checkpoint. All housekeeping is best-effort with try/except.

---

## Query-Time Search Filters

New in v0.5.1. Three filter types available at search time:

### Extension include (`-e` / `--ext`)

```bash
lss "auth" -e .py -e .ts
```

Applied in SQL: adds `AND (fts.file_path LIKE '%.py' OR fts.file_path LIKE '%.ts')` to the WHERE clause. This filters **before** BM25 scoring, so only matching files are scored and ranked. Very efficient.

### Extension exclude (`-E` / `--exclude-ext`)

```bash
lss "config" -E .json -E .yaml
```

Applied in SQL: adds `AND fts.file_path NOT LIKE '%.json' AND fts.file_path NOT LIKE '%.yaml'`. Also pre-scoring.

When both `-e` and `-E` are used, include is applied first, then exclude removes from the include set. E.g., `-e .py -e .ts -E .ts` → only `.py`.

### Content regex exclude (`-x` / `--exclude-pattern`)

```bash
lss "user data" -x '\d{4}-\d{2}-\d{2}'
```

Applied **post-scoring**: after BM25 + embedding + RRF + MMR produce the final ranked list, chunks whose snippet matches any regex pattern are removed. This is less efficient than SQL filtering (scoring happens on all candidates) but necessary because regex filtering requires the actual text content.

Multiple patterns combine with OR (a chunk is excluded if **any** pattern matches).

### Implementation detail

Extension filter SQL is built once by `_ext_filter_sql()` and injected into all SQL queries (initial BM25, PRF expansion). The `fts.file_path` column is `UNINDEXED` in FTS5 — it can't be used in MATCH but works fine in WHERE with LIKE.

Content regex patterns are compiled once in `semantic_search()` and passed as compiled `re.Pattern` objects. Invalid regex is caught early with a helpful error message.

---

## Search Algorithm Detail

### BM25 (Stage 1) — Custom Re-scoring

FTS5 does two jobs:
1. **Candidate retrieval** — fast inverted-index lookup via `MATCH` + `ORDER BY bm25(fts) LIMIT 600`
2. **Initial ordering** — FTS5's built-in `bm25()` provides a rough sort

After retrieval, we **re-score** candidates with our own BM25 implementation:

```
score(q, D) = Σ IDF(qi) · TF(qi, D) · (k1 + 1) / (TF(qi, D) + k1 · (1 - b + b · |D| / avgDL))
```

where:
- `IDF(qi) = log(1 + (N - df + 0.5) / (df + 0.5))` — inverse document frequency
- `TF(qi, D)` — raw term frequency in document D (counted from text)
- `|D|` — document length in tokens
- `avgDL` — average document length across the corpus
- `k1 = 1.2`, `b = 0.75` — standard BM25 parameters (tunable via `BM25_K1`, `BM25_B` env vars)
- `N` — total documents, `df` — document frequency (from `fts_vocab`)

**Why not use FTS5's built-in `bm25()`?** Its scores are extremely compressed on short passages — when every query term appears exactly once in a 200-word chunk, all candidates get nearly identical scores, destroying ranking quality. Our re-scorer uses proper TF saturation and document-length normalization with tunable parameters.

Pipeline:
1. Tokenize query → `["authentication", "jwt", "login"]`
2. Lookup document frequency for each term via `fts_vocab`
3. Filter high-DF terms (>15% of corpus) as stop words
4. Build FTS5 MATCH query: `"authentication" OR "login" OR "jwt"`
5. Execute with `ORDER BY bm25(fts) LIMIT 600` (candidate retrieval)
6. Scope filter: file_uid match (single file) or path prefix LIKE (directory)
7. **Re-score** candidates with custom BM25 (k1=1.2, b=0.75)

### PRF — Pseudo-Relevance Feedback (Stage 1.5)

Only triggered when:
- Query has <= 4 content tokens after DF filtering, OR
- BM25 top-10 scores are very flat (std < 0.02)

Process:
1. Extract TF-IDF weighted terms from top-10 BM25 results
2. Build expanded query: `(original) OR (anchor AND (expansions))`
3. Re-run FTS5 with expanded query, LIMIT 900
4. **Drift guard:** require >= 40% overlap with original top-10

### Embedding (Stage 2)

1. Collect top-28 BM25 results after Jaccard deduplication
2. Check SQLite `embeddings` table for cached vectors (batch SELECT)
3. Check in-memory LRU cache for remaining
4. **OpenAI:** Single API call: `[query] + [uncached_doc_texts]`
   **Local:** fastembed batch inference: `[query] + [uncached_doc_texts]`
5. Store new vectors in SQLite + LRU cache
6. Compute cosine similarity: `doc_vec @ query_vec`

### RRF Fusion (Stage 3)

For each document, compute:
```
rrf_score = 1/(K + bm25_rank) + 1/(K + embed_rank)
```
where K=60 (smoothing constant).

Then apply post-fusion boosts:
- Jaccard token overlap with query (40% weight)
- Phrase co-occurrence (30% weight)
- Digit matching (10% weight)

### MMR Re-ranking (Stage 3.5)

Only when vector coverage >= 90%:
```
mmr_score = 0.7 * relevance - 0.3 * max_similarity_to_already_selected
```

This prevents returning 5 nearly-identical chunks from the same file.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LSS_PROVIDER` | (auto-detect) | `openai` or `local` — embedding provider override |
| `LSS_DIR` | `~/.lss` | Data directory |
| `OPENAI_API_KEY` | (required for openai) | OpenAI API key for embeddings |
| `OPENAI_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_DIM` | `256` | OpenAI embedding dimensions |
| `LSS_MAX_FILE_SIZE` | `2097152` (2 MB) | Max file size to index |
| `OAI_TIMEOUT` | `2.0` | OpenAI API timeout (seconds) |
| `LSS_SPAN_WORDS` | `220` | Words per chunk |
| `LSS_SPAN_STRIDE` | `200` | Stride between chunks |
| `LSS_K_SQL` | `600` | BM25 candidate limit |
| `LSS_K_FINAL` | `20` | Final result pool size |
| `LSS_TOP_OAI` | `28` | Docs sent to embedding |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |
| `RRF_K` | `60` | RRF smoothing constant |
| `NO_COLOR` | (unset) | Disable ANSI colors |

### Config file (`~/.lss/config.json`)

```json
{
  "embedding_provider": "local",
  "watch_paths": ["/Users/me/Documents", "/Users/me/Projects"],
  "exclude_patterns": ["*.log", "*.min.js", "generated"],
  "include_extensions": [".rst", ".tex"]
}
```

---

## Performance Expectations

### Indexing throughput

| Scenario | Speed |
|----------|-------|
| Small project (50 files, 500 KB) | ~0.5s |
| Medium project (500 files, 5 MB) | ~4s |
| Large project (5000 files, 50 MB) | ~40s |
| Re-index (unchanged) | ~0.2 ms/file |

### Search latency

| Scenario | OpenAI | Local |
|----------|--------|-------|
| Cold (first search, no cache) | 400-800 ms | 50-200 ms |
| Warm (embeddings in SQLite) | 100-200 ms | 5-50 ms |
| Hot (all in LRU) | 50-150 ms | 2-10 ms |
| BM25-only (numeric query) | 1-5 ms | 1-5 ms |

### What makes it slow

1. **OpenAI API call** — 150-600 ms per call, unavoidable on cold search (OpenAI provider only)
2. **fastembed model load** — ~2s on first use (cached in memory after)
3. **Network latency** — API round-trip time (OpenAI provider only)
4. **DB writes during indexing** — SQLite FTS5 INSERT + WAL commit
5. **Large directories** — `os.walk` + filtering for thousands of files

### What makes it fast

1. **Content-addressed caching** — same content = same hash = cached embedding
2. **LRU in-memory cache** — repeated queries are instant
3. **Fast-path re-indexing** — stat() check, no content reading
4. **Inclusion-based filtering** — unknown extensions skipped without I/O
5. **Single batched API/inference call** — query + all docs in one request
6. **WAL mode** — readers never block writers, writers never block readers
7. **Local embeddings** — zero network dependency, ~50ms inference

---

## Project Layout

```
lss_config.py          Config: paths, env vars, provider detection, load/save
lss_extract.py         Document extractors: PDF, DOCX, XLSX, PPTX, HTML, EML, JSON, CSV
lss_store.py           Indexing: file discovery, inclusion filtering, smart chunking, FTS5
lss_cli.py             CLI: search, index, status, config, watch, include, exclude, eval, update
lss_sync.py            File watcher daemon (watchdog + debounced indexing)
semantic_search.py     Search: BM25, dual embedding providers, RRF, PRF, MMR
```

### Test suite — 366 tests

| Test file | Count | Description |
|---|---|---|
| `test_extract.py` | 39 | Document format extractors |
| `test_filtering.py` | 95 | Inclusion-based file filtering |
| `test_chunking.py` | 15 | Smart chunking (markdown/python/default) |
| `test_lss_store.py` | 14 | Storage layer |
| `test_lss_cli.py` | 21 | CLI unit tests |
| `test_e2e.py` | 25 | End-to-end workflows |
| `test_lss_sync.py` | 15 | File watcher daemon |
| `test_embedding_provider.py` | 23 | Provider detection, local embed, config |
| `test_cli_validation.py` | 92 | Full CLI surface area |
| `test_benchmark.py` | 11 | Performance benchmarks (OpenAI) |
| `test_search.py` | 6 | Search pipeline (OpenAI) |
| `test_search_quality.py` | 10 | Search quality metrics (OpenAI) |
| `test_beir.py` | 9 | BEIR benchmark adapter (excluded from default run) |

---

## Search Quality Benchmarks

### Evaluation Framework

LSS includes a comprehensive search quality evaluation framework in `tests/evaluation/`:

- **Golden set** (`golden_set.json`): 40 hand-labeled queries across 6 categories (keyword, conceptual, procedural, multi_concept, short_vague) with 3-level graded relevance (0/1/2) against a 33-file synthetic project corpus
- **BEIR adapter** (`beir_adapter.py`): Runs standard BEIR benchmark datasets through lss for comparison with published baselines
- **Evaluation harness** (`harness.py`): Orchestrates three-way comparison (BM25 / embedding / hybrid) using [ranx](https://github.com/AmenRa/ranx) for NDCG, MRR, Recall, Precision, MAP metrics

### Golden Set Results — OpenAI (text-embedding-3-small, 256d)

```
Method         NDCG@5  NDCG@10   MRR@10  Recall@5  Recall@10    P@5   P@10   MAP@10
bm25            0.870    0.885    0.988     0.845      0.893  0.480  0.255    0.809
embedding       0.844    0.886    0.988     0.788      0.917  0.445  0.265    0.784
hybrid          0.896    0.914    1.000     0.887      0.936  0.505  0.270    0.834
```

### Golden Set Results — Local (bge-small-en-v1.5, 384d)

```
Method         NDCG@5  NDCG@10   MRR@10  Recall@5  Recall@10    P@5   P@10   MAP@10
bm25            0.870    0.885    0.988     0.845      0.893  0.480  0.255    0.809
embedding       0.848    0.894    1.000     0.777      0.923  0.440  0.267    0.783
hybrid          0.888    0.911    1.000     0.864      0.931  0.495  0.268    0.834
```

Local embeddings are within **0.3%** of OpenAI on hybrid NDCG@10 — and **8x faster** (no network calls). BM25 scores are identical (same algorithm, same corpus).

Run this evaluation yourself: `lss eval` (uses current provider) or `LSS_PROVIDER=local lss eval`

### BEIR SciFact (5,183 biomedical docs, 300 queries)

| System | NDCG@10 |
|--------|---------|
| text-embedding-3-large | 0.735 |
| **lss hybrid** | **0.729** |
| Cohere embed-v3 | 0.717 |
| lss embedding-only | 0.719 |
| Voyage-2 | 0.713 |
| text-embedding-3-small | 0.694 |
| ColBERTv2 | 0.693 |
| BM25 (Anserini) | 0.665 |

lss hybrid beats ColBERTv2, Voyage-2, Cohere embed-v3, and text-embedding-3-small on SciFact.

### BEIR NFCorpus (3,633 medical docs, 323 queries)

| System | NDCG@10 |
|--------|---------|
| text-embedding-3-large | 0.361 |
| Cohere embed-v3 | 0.350 |
| lss embedding-only | 0.340 |
| ColBERTv2 | 0.338 |
| text-embedding-3-small | 0.336 |
| **lss hybrid** | **0.334** |
| BM25 (Anserini) | 0.325 |

Competitive with top systems. Note: hybrid slightly below embedding-only on biomedical text because FTS5 BM25's Porter stemmer doesn't handle medical terminology as well as Lucene's analyzer.

### Known Limitations

- **FTS5 Porter stemmer** is less capable than Lucene's analyzer on domain-specific vocabulary (medical, legal). This primarily affects BM25's contribution to hybrid search on specialized corpora.
- **Short passage scoring** — FTS5's built-in `bm25()` function produces flat scores when passages are short and uniform in length. Our custom BM25 re-scorer mitigates this.
- **Single-language** — FTS5 tokenizer is English-optimized. Non-Latin scripts may not tokenize well for BM25 (embedding search still works via multilingual models).
