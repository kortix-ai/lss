# LSS — Local Semantic Search

![Quick Demo](assets/quick_demo.gif)

Hybrid semantic search over local files. BM25 + embeddings fused with Reciprocal Rank Fusion. Real-time file watching. Runs 100% offline or with OpenAI — your choice.

```
lss "authentication JWT"              # search current directory
lss "deploy kubernetes" ~/Projects    # search a specific path
lss "rate limiting" --json            # machine-readable output
```

**0.91 NDCG@10** on our golden set. **Beats ColBERTv2, Voyage-2, and Cohere embed-v3** on BEIR SciFact. Works offline with local embeddings or with OpenAI for maximum quality. See [EVALS.md](EVALS.md) for full benchmarks.

---

## Install

```bash
# One-liner (auto-detects pipx/uv/pip)
curl -fsSL https://raw.githubusercontent.com/kortix-ai/lss/main/install.sh | bash
```

Or install directly:

```bash
pipx install local-semantic-search       # recommended — isolated install
pip install local-semantic-search         # classic
uv tool install local-semantic-search     # if you use uv
```

### Choose your embedding provider

**Option A: 100% offline (no API key needed)**

```bash
pip install 'local-semantic-search[local]'    # installs fastembed
lss config provider local                     # switch to local embeddings
```

Uses [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (384 dims, ~125 MB download on first run). No network calls, no API key, no cost.

**Option B: OpenAI embeddings (higher quality)**

```bash
export OPENAI_API_KEY="sk-..."   # add to ~/.zshrc or ~/.bashrc
```

Uses `text-embedding-3-small` (256 dims). Cached permanently — repeated searches cost zero API calls.

**Auto-detection:** If no provider is configured, lss checks for `OPENAI_API_KEY` first, then falls back to fastembed if installed.

---

## Usage

### Search

```bash
lss "Marko"                          # searches current directory
lss "Marko" ~/Documents              # explicit path (last arg if it exists on disk)
lss "Marko" -p ~/Documents           # explicit path with flag
lss "auth JWT" "deploy k8s"          # multiple queries
lss "database connection" --json     # JSON output for scripting
lss "config" -k 5                    # top 5 results
lss "error handling" | head          # pipe-friendly (colors auto-off)

# Query-time filters
lss "auth" -e .py -e .ts             # only search Python and TypeScript files
lss "config" -E .json -E .yaml       # exclude JSON and YAML files
lss "user data" -x '\d{4}-\d{2}-\d{2}'   # exclude chunks matching regex
lss "auth" -e .py -x "test_"         # combine: only .py, exclude test files
```

First search auto-indexes the directory. Subsequent searches use cached embeddings.

### Index

```bash
lss index ~/Projects                 # index without searching
lss index .                          # index current directory
lss index ~/Documents --yes          # skip confirmation prompt
```

### Manage

```bash
lss status                           # show DB stats, watched paths, provider, config
lss ls                               # list all indexed files
lss sweep --clear-all                # wipe the database

# Watch paths (for lss-sync daemon)
lss watch add ~/Documents
lss watch add ~/Projects
lss watch list
lss watch remove ~/Documents

# Exclude patterns
lss exclude add "*.log"
lss exclude add "*.min.js"
lss exclude list

# Include custom file extensions
lss include add .rst
lss include add .tex
lss include list
```

### Search Filters

Filter results at query time without re-indexing:

```bash
# Extension include (-e / --ext): only return results from these file types
lss "authentication" -e .py              # only Python files
lss "config" -e .yaml -e .toml -e .json  # only config files

# Extension exclude (-E / --exclude-ext): exclude these file types
lss "database" -E .sql                   # everything except SQL
lss "error" -E .log -E .txt             # skip logs and text

# Content regex exclude (-x / --exclude-pattern): filter out matching chunks
lss "user data" -x '\d{4}-\d{2}-\d{2}'  # exclude date patterns
lss "auth" -x "test_" -x "mock_"        # exclude test/mock code
lss "config" -x "(?i)deprecated"         # case-insensitive exclude

# Combine all filters
lss "authentication" -e .py -e .ts -x "test_" -x "fixture"
```

Extension filters are applied in SQL (efficient, pre-scoring). Content regex exclusion is applied post-scoring.

### Configuration

```bash
lss config show                      # display all configuration
lss config provider local            # switch to local embeddings
lss config provider openai           # switch to OpenAI embeddings
```

### File Watcher

```bash
lss-sync                             # watch paths from config
lss-sync --watch ~/Projects          # watch specific path
lss-sync --watch ~/a --watch ~/b     # multiple paths
```

Uses FSEvents (macOS) / inotify (Linux) to detect file changes and re-index in real time with debounced batching.

### Evaluate

```bash
lss eval                             # run search quality evaluation
lss eval --json                      # machine-readable
```

### Update

```bash
lss update                           # check for new version and upgrade
```

---

## How It Works

```
query "JWT authentication"
        |
   ┌────┴────┐
   v          v
  BM25    Embedding
(FTS5 +   (OpenAI or
 custom    fastembed +
 rescore)  cosine sim)
   |          |
   └────┬─────┘
        v
  Reciprocal Rank Fusion
        |
  Post-fusion boosts
  (Jaccard, phrase, digit)
        |
  MMR re-ranking
  (diversity)
        |
     results
```

1. **BM25** — SQLite FTS5 retrieves candidates by keyword, then our custom BM25 re-scorer ranks them with proper TF saturation and IDF weighting (k1=1.2, b=0.75).
2. **Embedding** — Query and top documents are embedded via OpenAI `text-embedding-3-small` (256d) or local fastembed `bge-small-en-v1.5` (384d). Cached in SQLite + LRU — repeated searches hit zero API calls.
3. **RRF** — Reciprocal Rank Fusion merges both ranked lists. No score calibration needed.
4. **Boosts** — Jaccard overlap, phrase matching, and digit co-mention features fine-tune ordering.
5. **MMR** — Maximal Marginal Relevance removes near-duplicate chunks for diverse results.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline with timing data.

---

## Document Formats

LSS indexes all common text, code, and document formats:

| Category | Extensions |
|----------|-----------|
| **Code** | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, `.php`, `.swift`, `.kt`, and 40+ more |
| **Markup** | `.md`, `.rst`, `.tex`, `.html`, `.xml`, `.yaml`, `.json`, `.toml` |
| **Documents** | `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.html`, `.eml` |
| **Data** | `.csv`, `.jsonl`, `.tsv` |
| **Config** | `.env.example`, `.gitignore`, `.dockerignore`, `Makefile`, `Dockerfile` |

Document extraction uses lightweight per-format libraries (pdfminer.six, python-docx, openpyxl, python-pptx, beautifulsoup4). All are optional — if a library isn't installed, that format is skipped silently.

Unknown extensions are **skipped by default**. Add custom extensions with `lss include add .ext`.

---

## Smart Chunking

Files are chunked intelligently based on file type:

| File type | Strategy | Split boundaries |
|-----------|----------|-----------------|
| **Markdown** (`.md`) | Heading-aware | Splits on `# heading` lines, preserving document structure |
| **Python** (`.py`) | Definition-aware | Splits on `def`/`class` boundaries, keeping functions intact |
| **Everything else** | Word-window | 220 words per chunk, 200-word stride (20-word overlap) |

Each chunk gets an MD5 hash (for dedup and embedding cache), an FTS5 entry (for BM25), and a lazily-computed embedding vector.

---

## Search Quality

### Golden Set (40 queries, 33-file project corpus)

**OpenAI embeddings** (`text-embedding-3-small`, 256d):

```
Method       NDCG@10   MRR@10   Recall@10
───────────────────────────────────────────
hybrid         0.914    1.000       0.936
embedding      0.886    0.988       0.917
bm25           0.885    0.988       0.893
```

**Local embeddings** (`bge-small-en-v1.5`, 384d):

```
Method       NDCG@10   MRR@10   Recall@10
───────────────────────────────────────────
hybrid         0.911    1.000       0.931
embedding      0.894    1.000       0.923
bm25           0.885    0.988       0.893
```

Local embeddings are within **0.3%** of OpenAI on NDCG@10 — and **8x faster** (no network calls).

### BEIR SciFact (5,183 docs, 300 queries) — NDCG@10

```
lss hybrid                  0.729
Cohere embed-v3             0.717
Voyage-2                    0.713
text-embedding-3-small      0.694
ColBERTv2                   0.693
BM25 (Anserini)             0.665
```

Full results and methodology: [EVALS.md](EVALS.md)

---

## Performance

| Scenario | OpenAI | Local |
|----------|--------|-------|
| Cold search (first query, no cache) | 400-800 ms | 50-200 ms |
| Warm search (embeddings cached) | 100-200 ms | 50-150 ms |
| Hot search (all in LRU memory) | 50-150 ms | 30-100 ms |
| Re-index unchanged files | 0.2 ms/file | 0.2 ms/file |
| Index 500 files | ~4s | ~4s |

With OpenAI, the API call is the bottleneck on cold search. With local embeddings, everything runs on your machine — no network dependency.

---

## File Filtering

LSS uses an **inclusion-based** approach — only known text/code/document extensions are indexed:

1. **~80 known extensions** — code, markup, config, documents (see `INDEXED_EXTENSIONS` in source)
2. **Known extensionless files** — `Makefile`, `Dockerfile`, `LICENSE`, `README`, etc.
3. **Directory exclusions** — `node_modules/`, `.git/`, `__pycache__/`, `.venv/`, `dist/`, `build/`, and ~70 more
4. **`.gitignore` parsing** — respects `.gitignore` patterns in every subtree
5. **User config** — `lss exclude add "*.log"` and `lss include add .ext`
6. **Max file size** — 2 MB default (override with `LSS_MAX_FILE_SIZE`)

Unknown extensions are skipped by default. This prevents indexing binary blobs, generated files, and other junk.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LSS_PROVIDER` | (auto-detect) | `openai` or `local` — embedding provider |
| `OPENAI_API_KEY` | — | OpenAI API key (required for openai provider) |
| `OPENAI_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_DIM` | `256` | OpenAI embedding dimensions |
| `LSS_DIR` | `~/.lss` | Data directory |
| `LSS_MAX_FILE_SIZE` | `2097152` (2 MB) | Max file size to index |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |
| `RRF_K` | `60` | RRF smoothing constant |
| `NO_COLOR` | (unset) | Disable ANSI colors |

### Config file (`~/.lss/config.json`)

```json
{
  "embedding_provider": "local",
  "watch_paths": ["/home/user/Documents", "/home/user/Projects"],
  "exclude_patterns": ["*.log", "*.min.js", "generated"],
  "include_extensions": [".rst", ".tex"]
}
```

---

## Programmatic Use

```python
from semantic_search import semantic_search
from lss_store import ingest_many, discover_files

# Index a directory
all_files, new_files, _ = discover_files("/path/to/project")
ingest_many(new_files)

# Search
results = semantic_search("/path/to/project", ["JWT authentication"])
for hit in results[0]:
    print(f"  {hit['score']:.3f}  {hit['file']}  {hit['text'][:80]}")
```

---

## Project Layout

```
lss_config.py          Config: paths, env vars, provider detection, load/save
lss_extract.py         Document extractors: PDF, DOCX, XLSX, PPTX, HTML, EML, JSON, CSV
lss_store.py           Indexing: file discovery, inclusion filtering, smart chunking, FTS5
lss_cli.py             CLI: search, index, status, config, watch, include, exclude, eval, update
lss_sync.py            File watcher daemon (watchdog + debounced indexing)
semantic_search.py     Search: BM25, dual embedding providers, RRF, PRF, MMR
ARCHITECTURE.md        Full technical pipeline reference
EVALS.md               Search quality benchmarks vs published systems
tests/                 366 tests (unit, e2e, benchmarks, search quality, BEIR)
```

## Tests

366 tests covering extraction, filtering, chunking, storage, CLI, e2e, file watching, embedding providers, and search quality.

```bash
python -m pytest tests/ -x -q             # run all tests
python -m pytest tests/ -k "not beir" -q   # skip BEIR (needs ir-datasets)
```

## License

Apache-2.0
