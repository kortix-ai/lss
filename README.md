# LSS — Local Semantic Search

![Quick Demo](assets/quick_demo.gif)

Hybrid search over local files. BM25 + embeddings + Reciprocal Rank Fusion. Real-time file watching. Runs offline or with OpenAI.

```
lss "authentication JWT"              # search current directory
lss "deploy kubernetes" ~/Projects    # search a specific path
lss "rate limiting" --json            # machine-readable output
```

**0.91 NDCG@10** on our golden set. Beats ColBERTv2, Voyage-2, and Cohere embed-v3 on BEIR SciFact. See [EVALS.md](EVALS.md).

---

## Install

```bash
# One-liner (auto-detects pipx/uv/pip)
curl -fsSL https://raw.githubusercontent.com/kortix-ai/lss/main/install.sh | bash
```

Or directly:

```bash
pipx install local-semantic-search       # recommended
pip install local-semantic-search
uv tool install local-semantic-search
```

### Embedding provider

**Default: OpenAI** — if `OPENAI_API_KEY` is set, lss uses it automatically.

```bash
export OPENAI_API_KEY="sk-..."   # add to ~/.zshrc or ~/.bashrc
```

**Offline alternative:**

```bash
pip install 'local-semantic-search[local]'
lss config provider local
```

Uses bge-small-en-v1.5 (384d, ~125 MB). No API key, no network, no cost. Within 0.3% of OpenAI on quality, 8x faster.

---

## Usage

### Search

```bash
lss "query"                              # current directory
lss "query" ~/Documents                  # explicit path
lss "auth JWT" "deploy k8s"              # multiple queries
lss "config" --json                      # JSON output
lss "error" -k 5                         # top 5

# Filters (applied without re-indexing)
lss "auth" -e .py -e .ts                 # only these extensions
lss "config" -E .json -E .yaml           # exclude extensions
lss "user data" -x '\d{4}-\d{2}-\d{2}'  # exclude chunks matching regex
lss "auth" -e .py -x "test_"            # combine filters
```

First search auto-indexes. Subsequent searches use cached embeddings.

### Index & manage

```bash
lss index ~/Projects                     # index without searching
lss status                               # DB stats, provider, config
lss ls                                   # list indexed files
lss sweep --clear-all                    # wipe database

lss watch add ~/Documents                # for lss-sync daemon
lss exclude add "*.log"                  # glob exclusion
lss include add .rst                     # custom extension
lss config provider local                # switch provider
lss eval                                 # run quality benchmarks
lss update                               # check for updates
```

### File watcher

```bash
lss-sync                                 # watch configured paths
lss-sync --watch ~/Projects              # watch specific path
```

---

## How it works

```
query → BM25 (FTS5 + custom rescore) ─┐
      → Embedding (OpenAI or local)  ──┤→ RRF → boosts → MMR → results
```

1. **BM25** — FTS5 retrieves candidates, custom re-scorer ranks with TF saturation + IDF (k1=1.2, b=0.75)
2. **Embedding** — OpenAI `text-embedding-3-small` (256d) or local `bge-small-en-v1.5` (384d), cached permanently
3. **RRF** — Reciprocal Rank Fusion merges both ranked lists
4. **Boosts** — Jaccard overlap, phrase matching, digit co-mention
5. **MMR** — Maximal Marginal Relevance for diversity

See [ARCHITECTURE.md](ARCHITECTURE.md) for full pipeline detail.

---

## Supported formats

| Category | Extensions |
|----------|-----------|
| Code | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, `.php`, `.swift`, `.kt`, +40 more |
| Markup | `.md`, `.rst`, `.tex`, `.html`, `.xml`, `.yaml`, `.json`, `.toml` |
| Documents | `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.eml` |
| Data | `.csv`, `.jsonl`, `.tsv` |

Extraction via pdfminer.six, python-docx, openpyxl, python-pptx, beautifulsoup4 (all optional — missing libs skip silently). Unknown extensions skipped by default; add with `lss include add .ext`.

---

## Search quality

| Method | NDCG@10 | MRR@10 | Provider |
|--------|---------|--------|----------|
| hybrid | 0.914 | 1.000 | OpenAI |
| hybrid | 0.911 | 1.000 | Local |
| bm25 | 0.885 | 0.988 | — |

BEIR SciFact (5,183 docs, 300 queries):

| System | NDCG@10 |
|--------|---------|
| **lss hybrid** | **0.729** |
| Cohere embed-v3 | 0.717 |
| Voyage-2 | 0.713 |
| ColBERTv2 | 0.693 |
| BM25 (Anserini) | 0.665 |

Full results: [EVALS.md](EVALS.md)

---

## Performance

| Scenario | OpenAI | Local |
|----------|--------|-------|
| Cold search (no cache) | 400-800 ms | 50-200 ms |
| Warm (embeddings cached) | 100-200 ms | 5-50 ms |
| Hot (all in LRU) | 50-150 ms | 2-10 ms |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `LSS_PROVIDER` | auto-detect | `openai` or `local` |
| `LSS_DIR` | `~/.lss` | Data directory |
| `BM25_K1` / `BM25_B` | 1.2 / 0.75 | BM25 tuning |
| `NO_COLOR` | unset | Disable ANSI colors |

Config file: `~/.lss/config.json`

---

## Programmatic use

```python
from semantic_search import semantic_search
from lss_store import ingest_many, discover_files

all_files, new_files, _ = discover_files("/path/to/project")
ingest_many(new_files)
results = semantic_search("/path/to/project", ["JWT authentication"])
```

---

## Tests

361+ tests covering extraction, filtering, chunking, CLI, e2e, file watching, providers, and search quality.

```bash
python -m pytest tests/ -x -q
```

## License

Apache-2.0
