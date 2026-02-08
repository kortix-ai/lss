---
name: kortix-semantic-search
description: "Full semantic search over Desktop files, agent memory, and knowledge. Use when the agent needs to find relevant files or search knowledge semantically."
---

# Semantic Search

You have a **full semantic search engine** running on this machine, powered by [lss](https://github.com/kortix-ai/lss) (Local Semantic Search). It indexes text files, code, PDFs, DOCX, XLSX, PPTX, HTML, EML, JSON, CSV — virtually everything.

A background file-watcher daemon (`lss-sync`) detects file changes in real time via FSEvents/inotify and re-indexes within seconds.

## How It Works

lss combines **BM25 full-text search** (keyword matching with custom re-scoring) and **embedding similarity** (semantic meaning via OpenAI or local fastembed) using Reciprocal Rank Fusion. Queries can be natural language — you don't need exact keywords.

The database lives at `/config/.lss/lss.db`.

## What's Indexed

| Source | Path | Content |
|--------|------|---------|
| **Desktop files** | `/config/Desktop` | Code, docs, PDFs, DOCX, XLSX, PPTX, HTML, JSON, CSV — all text-like files recursively |
| **Agent memory** | `/config/workspace/.kortix/` | MEMORY.md, memory/*.md, journal/*.md, knowledge/*.md |

Indexed formats: ~80 known extensions (code, markup, config, documents). Unknown extensions are skipped. `.gitignore` patterns are respected.

## Quick Reference

```bash
# Search EVERYTHING indexed
lss "your natural language query" -p /config/Desktop -k 10 --json

# Search only agent memory + knowledge
lss "user deployment preferences" -p /config/workspace/.kortix/ -k 5 --json

# Search a specific project directory
lss "database migration strategy" -p /config/Desktop/myproject/ -k 5 --json

# Search without triggering re-indexing (faster, uses existing index)
lss "query" -p /config/Desktop --no-index -k 10 --json

# Filter by file type
lss "auth logic" -p /config/Desktop -e .py -e .ts -k 10 --json

# Exclude file types
lss "config" -p /config/Desktop -E .json -E .yaml -k 10 --json

# Exclude content patterns
lss "user data" -p /config/Desktop -x '\d{4}-\d{2}-\d{2}' -k 10 --json

# Force re-index a path immediately
lss index /config/Desktop/important-file.md

# List all indexed files
lss ls

# Check index stats and configuration
lss status
```

## Search Filters

Narrow results at query time without re-indexing.

| Flag | Meaning | Applied | Example |
|------|---------|---------|---------|
| `-e EXT` / `--ext` | Include only these extensions (repeatable) | In SQL, pre-scoring | `-e .py -e .ts` |
| `-E EXT` / `--exclude-ext` | Exclude these extensions (repeatable) | In SQL, pre-scoring | `-E .json -E .yaml` |
| `-x REGEX` / `--exclude-pattern` | Exclude chunks matching regex (repeatable) | Post-scoring | `-x 'test_' -x 'TODO'` |

### Filter Strategy for Agents

**Narrow first, then broaden.** Start with tight extension filters, then remove them if results are insufficient:

```bash
# 1. Try narrow: only Python source files
lss "authentication flow" -p /config/Desktop/project -e .py -k 10 --json

# 2. If too few results, broaden to all code
lss "authentication flow" -p /config/Desktop/project -e .py -e .ts -e .go -e .rs -k 10 --json

# 3. If still insufficient, search everything
lss "authentication flow" -p /config/Desktop/project -k 10 --json
```

**Exclude noise, not signal:**

```bash
# Exclude generated/test code when looking for implementations
lss "rate limiting" -e .py -x "test_" -x "mock_" -x "fixture" --json -k 10

# Exclude logs and config when looking for code
lss "database connection" -E .log -E .yaml -E .json -E .toml --json -k 10

# Exclude dates/timestamps from data searches
lss "customer report" -x '\d{4}-\d{2}-\d{2}' --json -k 10
```

## JSON Output Format

**Always use `--json` for programmatic parsing.**

```bash
lss "query" -p /config/Desktop --json -k 10
```

Returns an array of result arrays (one per query):
```json
[
  {
    "query": "authentication flow",
    "hits": [
      {
        "file_path": "/config/Desktop/project/auth.py",
        "score": 0.0345,
        "snippet": "def authenticate(user, password):\n    \"\"\"Authenticate user with JWT...",
        "rank_stage": "S3_MMR",
        "indexed_at": 1738900000.0
      }
    ]
  }
]
```

Key fields:
- `file_path` — Full path to the source file
- `score` — Relevance score (higher is better)
- `snippet` — Best-matching text excerpt (~280 chars)
- `rank_stage` — S1=BM25 only, S3=fusion, S3_MMR=fusion+diversity

## Multi-Query Decomposition

For complex questions, **decompose into multiple specific queries**:

```bash
# BAD: single vague query
lss "how does the system work" --json -k 10

# GOOD: decomposed into specific queries
lss "system architecture overview" "API endpoint design" "database schema" "authentication flow" --json -k 5
```

Or use a query file:
```bash
echo "system architecture overview" > /tmp/queries.txt
echo "API endpoint design" >> /tmp/queries.txt
echo "database schema" >> /tmp/queries.txt
lss -Q /tmp/queries.txt -p /config/Desktop/project --json -k 5
```

## When to Use lss vs grep

| Use lss | Use grep |
|---------|----------|
| Conceptual queries ("how to handle errors") | Exact string ("ERROR_CODE_429") |
| Fuzzy matching ("something like the email template") | Specific variable name (`userSessionToken`) |
| Cross-file discovery ("files about API design") | Known file + line search |
| Natural language ("what's the deploy process") | Regex pattern matching |

## Iterative Search Strategy

For large codebases, use an iterative approach:

```bash
# Step 1: Broad discovery — find relevant areas
lss "payment processing" -p /config/Desktop/project -k 20 --json

# Step 2: Narrow by extension — focus on implementation
lss "payment processing" -p /config/Desktop/project -e .py -k 10 --json

# Step 3: Narrow by path — focus on specific module
lss "payment processing" -p /config/Desktop/project/src/payments/ -k 10 --json

# Step 4: Read the actual files for full context
cat /config/Desktop/project/src/payments/processor.py
```

## Indexing

```bash
# Index a directory (auto-triggered on first search)
lss index /config/Desktop/project/

# Index a single file
lss index /config/Desktop/important.pdf

# The daemon handles real-time updates automatically
# Use manual indexing only for immediate needs
```

## Maintenance

```bash
# Sweep stale entries (files deleted from disk)
lss sweep --retention-days 90

# Clear all embeddings (forces re-embedding on next search)
lss sweep --clear-embeddings 0

# Full reset
lss sweep --clear-all
```

## Rules

1. **Always use `--json` flag** when searching programmatically. Parse the JSON output.
2. **Always use `-p <path>`** to scope searches. Never search without a path scope.
3. **Use `-k` to control result count.** `-k 5` for focused, `-k 20` for broad exploration.
4. **Decompose complex queries.** Multiple specific queries beat one vague query.
5. **Read source files for full context.** Snippets are excerpts — always read the full file after finding a match.
6. **Narrow with filters first.** Use `-e` to target file types before broadening.
7. **Don't grep when lss is better.** Conceptual, fuzzy, and cross-file queries belong in lss.
8. **The index auto-updates.** Only use `lss index` for immediate needs. The daemon handles the rest.
