# LSS Architecture & Pipeline Reference

> Local Semantic Search — how it works, step by step, with real timing data.

---

## Overview

LSS finds content in your files by combining two complementary search strategies:

1. **BM25** (keyword match via SQLite FTS5) — fast, exact, zero-cost
2. **OpenAI embeddings** (semantic similarity) — understands meaning, costs API calls

Results from both are merged using **Reciprocal Rank Fusion (RRF)**, then re-ranked with **MMR** (Maximal Marginal Relevance) for diversity.

```
user query
    |
    v
[INDEXING PIPELINE]          [SEARCH PIPELINE]
file discovery               tokenize + DF lookup
    |                            |
text detection               keyword extraction
    |                            |
read + extract               FTS5 BM25 query
    |                            |
normalize                    PRF expansion (optional)
    |                            |
chunk (220-word spans)       OpenAI embedding API  <-- BOTTLENECK
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
| 1 | **File discovery** | `os.walk` + in-place dir pruning + filter excluded dirs/extensions/names | 16.0 ms | 25% |
| 2 | **Text detection** | Extension fast-path → byte-level check (read 8KB) | 0.2 ms | <1% |
| 3 | **Read + extract** | UTF-8 read, JSON/CSV/JSONL parsing, PDF via PyPDF2 | 0.7 ms | 1% |
| 4 | **Normalize** | Unicode NFKC + whitespace collapse | 1.1 ms | 2% |
| 5 | **Chunk** | Sliding window: 220 words/span, 200-word stride | 0.2 ms | <1% |
| 6 | **Hash** | MD5 content signature (file) + MD5 per chunk | 0.5 ms | 1% |
| 7 | **DB write** | FTS5 INSERT + files manifest + COMMIT | 45.6 ms | **71%** |
| | **TOTAL** | | **64.1 ms** | |
| | **Re-index (cached)** | Fast path: stat() + LRU cache check | 1.8 ms | |

### Key insight — Indexing

**DB writes dominate indexing (71%).** The actual text processing (read, normalize, chunk, hash) takes <5 ms combined. SQLite FTS5 INSERT + WAL commit is the bottleneck. This is acceptable because:

- Indexing happens once per file (content-addressed via MD5)
- Re-indexing unchanged files hits the LRU fast path (~0.2 ms/file)
- `ingest_many` batches all files in a single transaction

### Re-index fast path

When a file hasn't changed (same path + size + mtime + version key):
1. Check in-memory LRU cache (O(1) hash lookup) → **0.01 ms**
2. If not in LRU, check SQLite `files` table with size/mtime guard → **0.2 ms**
3. No content hashing, no text extraction, no FTS writes

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
| 6 | **OpenAI embedding** | Single API call: [query] + [top-28 doc chunks] | **565** | **0.1*** | **0.1*** |
| 7 | **Embedding cache** | Write new vectors to SQLite `embeddings` table | 1-5 | 0 | 0 |
| 8 | **RRF fusion** | Reciprocal Rank Fusion of BM25 + embedding ranks | <0.1 | <0.1 | <0.1 |
| 9 | **Post-fusion boost** | Jaccard, phrase, digit co-mention features | <0.1 | <0.1 | <0.1 |
| 10 | **MMR re-ranking** | Vector-MMR (lambda=0.7) for diversity, if coverage >= 90% | <0.1 | <0.1 | <0.1 |
| | **TOTAL** | | **~570** | **~157** | **~122** |

*\* Warm/hot: embeddings served from SQLite cache or in-memory LRU*

### Key insight — Search

**The OpenAI embedding API call dominates cold search (90%+ of wall time).** Everything else — BM25, fusion, MMR — takes <2 ms combined. The caching strategy is critical:

| Cache layer | Scope | TTL | Lookup cost |
|-------------|-------|-----|-------------|
| **LRU (in-memory)** | Per-process, query + doc vectors | 15 min (query), 60 min (doc) | ~0.001 ms |
| **SQLite `embeddings` table** | Persistent, doc vectors only | Forever (until sweep) | ~0.1 ms |
| **OpenAI API** | N/A | N/A | **150-600 ms** |

### Three thermal states

- **Cold** (~570 ms): First search ever, no caches. Hits OpenAI API for query + all doc embeddings.
- **Warm** (~157 ms): Doc embeddings cached in SQLite. Only query embedding hits API (but even that may be cached in LRU if same query repeated).
- **Hot** (~122 ms): Everything in LRU. Zero API calls. Pure local compute.

After a directory is searched once, subsequent searches are warm or hot.

---

## File Filtering — What Gets Indexed

LSS uses a three-layer filter to avoid indexing junk:

### Layer 1: Directory exclusions (`EXCLUDED_DIRS`)

Entire directory trees are pruned in-place during `os.walk`. This is the most impactful filter — it prevents walking into `node_modules/` (which can have 100K+ files).

```
.git, node_modules, __pycache__, .venv, venv, dist, build, target,
.next, .nuxt, .idea, .vscode, coverage, .cache, .terraform, .turbo,
.gradle, .mvn, bin, obj, packages, site-packages, logs, .lss, ...
```

Full list: ~70 directory names. See `EXCLUDED_DIRS` in `lss_store.py`.

### Layer 2: File name/extension exclusions

**Binary extensions** (`BINARY_EXTENSIONS`): ~100 extensions that are always binary — skipped without reading any bytes. Images, video, audio, archives, compiled code, fonts, ML models, etc.

**Excluded file names** (`EXCLUDED_FILES`): Lock files, env files, OS junk:
```
package-lock.json, yarn.lock, pnpm-lock.yaml, poetry.lock,
Cargo.lock, go.sum, .DS_Store, .env, .env.local, ...
```

**Max file size** (`LSS_MAX_FILE_SIZE`): 2 MB default. Files larger than this are skipped. Override with `LSS_MAX_FILE_SIZE=10485760` (10 MB).

### Layer 3: Content-based detection

For files that pass layers 1-2, `_is_text_file()` reads the first 8 KB and checks:
1. No null bytes (binary indicator)
2. Valid UTF-8
3. Fallback: Latin-1 with >70% printable characters

### Layer 4: User-configured exclusions

`lss exclude add <pattern>` adds glob patterns to `~/.lss/config.json`:
```
lss exclude add "*.log"
lss exclude add "*.min.js"
lss exclude add "generated"
lss exclude add "*.snap"
```

These are checked via `fnmatch` against file names and relative paths.

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
    version     TEXT NOT NULL,     -- "text-embedding-3-small:256:p2:c4"
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
    model     TEXT,        -- "text-embedding-3-small"
    dim       INTEGER,     -- 256
    version   TEXT,
    vector    BLOB,        -- float32 array, 256 * 4 = 1024 bytes
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

## Chunking Strategy

Files are chunked into overlapping spans:

- **Span size:** 220 words (~1 paragraph of dense text)
- **Stride:** 200 words (20-word overlap between consecutive spans)
- **Why overlapping:** Ensures no sentence falls on a boundary and gets missed

A 1000-word file produces ~5 chunks. A 10,000-word file produces ~50 chunks.

Each chunk gets:
- An MD5 hash (for deduplication and embedding cache lookup)
- An FTS5 entry (for BM25 keyword search)
- An embedding vector (computed lazily on first search, cached permanently)

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
4. Single OpenAI API call: `[query] + [uncached_doc_texts]`
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
| `LSS_DIR` | `~/.lss` | Data directory |
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings |
| `OPENAI_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_DIM` | `256` | Embedding dimensions |
| `LSS_MAX_FILE_SIZE` | `2097152` (2 MB) | Max file size to index |
| `OAI_TIMEOUT` | `2.0` | OpenAI API timeout (seconds) |
| `LSS_SPAN_WORDS` | `220` | Words per chunk |
| `LSS_SPAN_STRIDE` | `200` | Stride between chunks |
| `LSS_K_SQL` | `600` | BM25 candidate limit |
| `LSS_K_FINAL` | `20` | Final result pool size |
| `LSS_TOP_OAI` | `28` | Docs sent to embedding API |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |
| `RRF_K` | `60` | RRF smoothing constant |
| `NO_COLOR` | (unset) | Disable ANSI colors |

### Config file (`~/.lss/config.json`)

```json
{
  "watch_paths": ["/Users/me/Documents", "/Users/me/Projects"],
  "exclude_patterns": ["*.log", "*.min.js", "generated"]
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

| Scenario | Latency |
|----------|---------|
| Cold (first search, no cache) | 400-800 ms |
| Warm (embeddings in SQLite) | 100-200 ms |
| Hot (all in LRU) | 50-150 ms |
| BM25-only (numeric query) | 1-5 ms |

### What makes it slow

1. **OpenAI API call** — 150-600 ms per call, unavoidable on cold search
2. **Network latency** — API round-trip time dominates
3. **DB writes during indexing** — SQLite FTS5 INSERT + WAL commit
4. **Large directories** — `os.walk` + `_is_text_file` for thousands of files

### What makes it fast

1. **Content-addressed caching** — same content = same hash = cached embedding
2. **LRU in-memory cache** — repeated queries are instant
3. **Fast-path re-indexing** — stat() check, no content reading
4. **Extension-based filtering** — binary files skipped without I/O
5. **Single batched API call** — query + all docs in one request
6. **WAL mode** — readers never block writers, writers never block readers

---

## Search Quality Benchmarks

### Evaluation Framework

LSS includes a comprehensive search quality evaluation framework in `tests/evaluation/`:

- **Golden set** (`golden_set.json`): 40 hand-labeled queries across 6 categories (keyword, conceptual, procedural, multi_concept, short_vague) with 3-level graded relevance (0/1/2) against a 30-file synthetic project corpus
- **BEIR adapter** (`beir_adapter.py`): Runs standard BEIR benchmark datasets through lss for comparison with published baselines
- **Evaluation harness** (`harness.py`): Orchestrates three-way comparison (BM25 / embedding / hybrid) using [ranx](https://github.com/AmenRa/ranx) for NDCG, MRR, Recall, Precision, MAP metrics

### Golden Set Results (40 queries, 30-file corpus)

```
Method         NDCG@5  NDCG@10   MRR@10  Recall@5  Recall@10    P@5   P@10   MAP@10
bm25            0.885    0.901    0.988     0.857      0.906  0.485  0.258    0.822
embedding       0.860    0.901    0.988     0.801      0.930  0.450  0.267    0.796
hybrid          0.910    0.929    1.000     0.893      0.948  0.505  0.273    0.846
```

With custom BM25 re-scoring (v0.4.0), BM25-only search matches embedding-only on NDCG@10. The hybrid fusion still wins across all metrics.

Run this evaluation yourself: `lss eval`

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
- **Single-language** — FTS5 tokenizer is English-optimized. Non-Latin scripts may not tokenize well for BM25 (embedding search still works via OpenAI's multilingual models).
