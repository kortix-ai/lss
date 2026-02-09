# LSS Architecture

## Pipeline

```
[INDEXING]                    [SEARCH]
os.walk + inclusion filter    tokenize + DF lookup
  → extract (PDF/DOCX/etc)      → keyword extraction
  → normalize (NFKC)            → FTS5 BM25 + custom rescore
  → smart chunk                  → PRF expansion (optional)
  → MD5 hash                    → embed (OpenAI or local)
  → FTS5 insert                 → RRF fusion + boosts
                                 → MMR re-rank
```

---

## Indexing

| Step | What | Time (8 files, 59KB) |
|------|------|---------------------|
| File discovery | `os.walk` + dir pruning + inclusion filter + .gitignore | 16 ms |
| Extraction | Plain text or PDF/DOCX/XLSX/PPTX/HTML/EML/JSON/CSV dispatch | 0.7 ms |
| Normalize + chunk | NFKC, then markdown/python/word-window chunking | 1.3 ms |
| Hash | MD5 per file + per chunk | 0.5 ms |
| DB write | FTS5 INSERT + WAL COMMIT | **45.6 ms** |
| **Total** | | **64.1 ms** |
| Re-index (unchanged) | stat() + LRU check | 1.8 ms |

DB writes dominate (71%). Text processing is <5 ms. Acceptable because indexing is once per file (content-addressed via MD5), and re-index hits an LRU fast path at ~0.2 ms/file.

### Extraction (`lss_extract.py`)

| Format | Library | Extracted |
|--------|---------|-----------|
| PDF | pdfminer.six | Layout-aware text, all pages |
| DOCX | python-docx | Paragraphs + table cells |
| XLSX | openpyxl | All sheets, tab-separated rows |
| PPTX | python-pptx | Text frames + tables |
| HTML | beautifulsoup4 | Visible text (no scripts/styles) |
| EML | stdlib email | Subject + From + plain body |
| JSON/JSONL/CSV | stdlib | Pretty-printed / header:value pairs |

All return `""` on error. Missing libraries skip silently.

### Chunking (`lss_store.py`)

| Type | Strategy |
|------|----------|
| `.md` | Split on `# heading` lines, merge small sections |
| `.py` | Split on `def`/`class` boundaries |
| Everything else | 220-word window, 200-word stride (20-word overlap) |

### File filtering

Seven layers, inclusion-based:

1. **Directory exclusions** — `node_modules/`, `.git/`, `__pycache__/`, etc. (~70 names, pruned during walk)
2. **`.gitignore` parsing** — per-subtree
3. **`INDEXED_EXTENSIONS`** — ~80 known text/code/doc extensions
4. **`KNOWN_EXTENSIONLESS`** — `Makefile`, `Dockerfile`, `LICENSE`, etc.
5. **Excluded filenames** — `package-lock.json`, `yarn.lock`, `.DS_Store`, etc.
6. **Max file size** — 2 MB (override: `LSS_MAX_FILE_SIZE`)
7. **User config** — `lss include add` / `lss exclude add`

Unknown extensions skipped by default.

---

## Search

| Step | What | Cold (ms) | Warm | Hot |
|------|------|----------|------|-----|
| Tokenize + DF lookup | Regex → `fts_vocab` batch query | 0.2 | 0.2 | 0.1 |
| Keyword extraction | DF filter, phrase detect | 0.1 | 0.1 | 0.1 |
| FTS5 BM25 | `MATCH ? ORDER BY bm25(fts) LIMIT 600` | 0.4 | 0.4 | 0.3 |
| PRF expansion | Top-10 term extraction, re-query | 0-2 | 0-2 | 0-1 |
| Embedding | OpenAI API / fastembed inference | **565 / 50** | 0.1 | 0.1 |
| RRF + boosts + MMR | Fusion, Jaccard/phrase, diversity | <0.3 | <0.3 | <0.3 |
| **Total (OpenAI)** | | **~570** | **~157** | **~122** |
| **Total (local)** | | **~55** | **~5** | **~2** |

Warm/hot: embeddings served from SQLite cache or LRU. OpenAI API dominates cold search (90%+). Local has no network dependency.

### BM25 custom re-scoring

FTS5's built-in `bm25()` produces flat scores on short uniform passages. We re-score with:

```
score(q, D) = sum IDF(qi) * TF(qi,D) * (k1+1) / (TF + k1*(1 - b + b*|D|/avgDL))
```

k1=1.2, b=0.75. This gave **4.4x NDCG improvement** over raw FTS5 scoring.

### PRF (Pseudo-Relevance Feedback)

Triggered when query has <=4 tokens or BM25 top-10 scores are flat. Extracts TF-IDF terms from top-10, re-queries with expansion. Drift guard requires >=40% overlap with original results.

### Embedding

Top-28 BM25 results → check SQLite cache → check LRU → embed uncached via OpenAI or fastembed → cache new vectors → cosine similarity.

### RRF fusion

```
rrf_score = 1/(60 + bm25_rank) + 1/(60 + embed_rank)
```

Post-fusion: Jaccard (40%), phrase (30%), digit (10%) boosts. MMR (lambda=0.7) when vector coverage >= 90%.

### Query-time filters

| Flag | Where applied | How |
|------|--------------|-----|
| `-e .py` (include ext) | SQL pre-scoring | `AND file_path LIKE '%.py'` |
| `-E .json` (exclude ext) | SQL pre-scoring | `AND file_path NOT LIKE '%.json'` |
| `-x 'regex'` (content exclude) | Post-scoring | Compiled regex on snippet text |

Extension SQL built by `_ext_filter_sql()`, injected into all queries.

### Caching

| Layer | Scope | Cost |
|-------|-------|------|
| LRU (in-memory) | Per-process, query+doc vectors | ~0.001 ms |
| SQLite `embeddings` | Persistent, doc vectors | ~0.1 ms |
| OpenAI API | N/A | 150-600 ms |
| fastembed inference | N/A | 30-80 ms |

### Provider detection

1. `LSS_PROVIDER` env var
2. `~/.lss/config.json` `embedding_provider`
3. `OPENAI_API_KEY` set → `"openai"`
4. `fastembed` importable → `"local"`
5. Fallback: `"openai"` (fails with helpful error)

`VERSION_KEY` includes provider — switching triggers re-embedding; BM25 index stays.

---

## Database

SQLite WAL mode, `~/.lss/lss.db`.

```sql
CREATE TABLE files (
    file_uid TEXT PRIMARY KEY, path TEXT, size INTEGER, mtime REAL,
    content_sig TEXT, version TEXT, indexed_at REAL, status TEXT DEFAULT 'active'
);

CREATE VIRTUAL TABLE fts USING fts5(
    id UNINDEXED, text, file_uid UNINDEXED, file_path UNINDEXED,
    text_hash UNINDEXED, tokenize='porter', prefix='2'
);

CREATE VIRTUAL TABLE fts_vocab USING fts5vocab(fts, row);

CREATE TABLE embeddings (
    text_hash BLOB, model TEXT, dim INTEGER, version TEXT, vector BLOB,
    created REAL DEFAULT (unixepoch()),
    PRIMARY KEY (text_hash, model, dim, version)
);
```

Locking: writers use PASSIVE checkpoints (30s busy timeout), readers never checkpoint. Embedding cache writes are best-effort with try/except.

---

## Project layout

```
lss_config.py       Config, provider detection, persistent config
lss_extract.py      Document extractors (PDF, DOCX, XLSX, PPTX, HTML, EML, JSON, CSV)
lss_store.py        File discovery, filtering, chunking, FTS5 indexing
lss_cli.py          CLI (search, index, status, config, watch, eval, update)
lss_sync.py         File watcher daemon (watchdog + debounced batching)
semantic_search.py  Search pipeline (BM25, embedding, RRF, PRF, MMR)
```
