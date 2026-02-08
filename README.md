# LSS — Local Semantic Search

Hybrid semantic search over local files. BM25 + OpenAI embeddings fused with Reciprocal Rank Fusion. Real-time file watching. Runs on any machine.

```
lss "authentication JWT"              # search current directory
lss "deploy kubernetes" ~/Projects    # search a specific path
lss "rate limiting" --json            # machine-readable output
```

**0.93 NDCG@10** on our golden set. **Beats ColBERTv2, Voyage-2, and Cohere embed-v3** on BEIR SciFact. See [EVALS.md](EVALS.md) for full benchmarks.

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

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."   # add to ~/.zshrc or ~/.bashrc
```

That's it. No other dependencies, no GPU, no Docker.

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
lss status                           # show DB stats, watched paths, config
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

---

## How It Works

```
query "JWT authentication"
        |
   ┌────┴────┐
   v          v
  BM25    Embedding
(FTS5 +   (OpenAI API +
 custom    cosine sim)
 rescore)
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
2. **Embedding** — Query and top documents are embedded via `text-embedding-3-small` (256 dims). Cached in SQLite + LRU — repeated searches hit zero API calls.
3. **RRF** — Reciprocal Rank Fusion merges both ranked lists. No score calibration needed.
4. **Boosts** — Jaccard overlap, phrase matching, and digit co-mention features fine-tune ordering.
5. **MMR** — Maximal Marginal Relevance removes near-duplicate chunks for diverse results.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline with timing data.

---

## Search Quality

### Golden Set (40 queries, 30-file project corpus)

```
Method       NDCG@10   MRR@10   Recall@10
───────────────────────────────────────────
hybrid         0.932    1.000       0.948
bm25           0.888    0.971       0.895
embedding      0.901    0.988       0.930
```

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

| Scenario | Latency |
|----------|---------|
| Cold search (first query, no cache) | 400-800 ms |
| Warm search (embeddings cached in SQLite) | 100-200 ms |
| Hot search (all in LRU memory) | 50-150 ms |
| Re-index unchanged files | 0.2 ms/file |
| Index 500 files | ~4s |

The OpenAI API call is the bottleneck on cold search. After first search, everything is cached.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_DIM` | `256` | Embedding dimensions |
| `LSS_DIR` | `~/.lss` | Data directory |
| `LSS_MAX_FILE_SIZE` | `2097152` (2 MB) | Max file size to index |
| `BM25_K1` | `1.2` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |
| `RRF_K` | `60` | RRF smoothing constant |
| `NO_COLOR` | (unset) | Disable ANSI colors |

### Config file (`~/.lss/config.json`)

```json
{
  "watch_paths": ["/home/user/Documents", "/home/user/Projects"],
  "exclude_patterns": ["*.log", "*.min.js", "generated"]
}
```

---

## Programmatic Use

```python
from semantic_search import semantic_search
from lss_store import ingest_many, discover_files

# Index a directory
files = discover_files("/path/to/project")
ingest_many(files)

# Search
results = semantic_search("/path/to/project", ["JWT authentication"])
for hit in results[0]:
    print(f"  {hit['score']:.3f}  {hit['file']}  {hit['text'][:80]}")
```

---

## Project Layout

```
lss_config.py          Config: paths, env vars, load/save
lss_store.py           Indexing: file discovery, text extraction, FTS5 storage
lss_cli.py             CLI: search, index, status, watch, exclude, eval
lss_sync.py            File watcher daemon (watchdog + debounced indexing)
semantic_search.py     Search engine: BM25, embeddings, RRF, PRF, MMR
ARCHITECTURE.md        Full technical pipeline reference
EVALS.md               Search quality benchmarks vs published systems
tests/                 90 tests (unit, e2e, benchmarks, search quality, BEIR)
```

## License

Apache-2.0
