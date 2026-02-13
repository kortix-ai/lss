# Developer Workflows

Day-to-day development, testing, and release procedures for lss.

---

## Setup

```bash
# Clone
git clone https://github.com/kortix-ai/lss.git && cd lss

# Create venv + install (editable with dev deps)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Or with uv:
uv venv && uv sync -e dev
```

Set your API key (needed for embedding tests and OpenAI provider):

```bash
export OPENAI_API_KEY="sk-..."
```

For 100% offline development, switch to local embeddings:

```bash
pip install -e ".[dev,local]"
lss config provider local
```

---

## Running locally

```bash
# Via the installed entry point
lss "authentication JWT"
lss status

# Or directly (no install needed)
python -m lss_cli "authentication JWT"
python lss_cli.py "authentication JWT"
```

---

## Testing

### Quick (no API calls)

```bash
pytest                                   # all offline tests (~9s)
pytest tests/test_lss_store.py -v        # just the storage layer
pytest tests/test_cli_validation.py -v   # CLI argument parsing
pytest -k "not beir and not e2e"         # skip heavy tests
```

### Full (requires OPENAI_API_KEY)

```bash
pytest tests/test_e2e.py -v             # end-to-end search
pytest tests/test_search.py -v          # search pipeline
pytest tests/test_search_quality.py -v  # quality assertions
```

### BEIR benchmarks (~10 min)

```bash
pytest tests/test_beir.py -v
lss eval                                # golden-set evaluation
lss eval --json                         # machine-readable
```

### Test isolation

Every test gets its own `LSS_DIR` (via the `isolated_lss_dir` fixture in
`tests/conftest.py`), so tests never share state and can run in parallel.

---

## Linting & formatting

```bash
ruff check .                            # lint
ruff check . --fix                      # auto-fix
black --check .                         # format check
black .                                 # auto-format
mypy .                                  # type check
```

---

## Project layout

```
lss_config.py       Config, provider detection, VERSION_KEY, persistent config
lss_extract.py      Document extractors (PDF, DOCX, XLSX, PPTX, HTML, EML, JSON, CSV)
lss_store.py        File discovery, filtering, chunking, FTS5 indexing, DB schema
lss_cli.py          CLI entry point (search, index, status, config, watch, eval, update)
lss_sync.py         File watcher daemon (watchdog + debounced batching)
semantic_search.py  Search pipeline (BM25, embedding, RRF, PRF, MMR)
__main__.py         python -m support
pyproject.toml      Package metadata, deps, tool config
tests/              Test suite
  conftest.py         Shared fixtures (isolated_lss_dir, sample_dir)
  test_lss_store.py   Storage layer tests
  test_lss_cli.py     CLI integration tests
  test_cli_validation.py  CLI argument/flag validation
  test_lss_sync.py    File watcher tests
  test_search_filters.py  Query-time filter tests
  test_chunking.py    Text chunking tests
  test_extract.py     Document extraction tests
  test_filtering.py   File filtering/exclusion tests
  test_e2e.py         End-to-end (requires API key)
  test_search.py      Search pipeline (requires API key)
  test_search_quality.py  Quality assertions (requires API key)
  test_beir.py        BEIR benchmark adapter
  test_benchmark.py   Performance benchmarks
  evaluation/         Golden-set evaluation harness
```

---

## Key concepts

### VERSION_KEY

Defined in `lss_config.py`:

```
VERSION_KEY = "{model}:{dim}:p{PREPROC_VER}:c{CHUNKER_VER}"
```

This key is stored with every indexed file. When you change the embedding
model/dimensions or bump `PREPROC_VER`/`CHUNKER_VER`, all files are
automatically re-indexed on next search.

**When to bump:**
- Changed preprocessing logic (normalization, text extraction) -> bump `PREPROC_VER`
- Changed chunking logic (span size, overlap, strategy) -> bump `CHUNKER_VER`
- Changed embedding model/dimensions -> happens automatically via config

### file_uid

Each file gets a unique ID based on its **resolved absolute path** (not content).
This ensures files with identical content (e.g. multiple `__init__.py`) each get
their own DB row. Content deduplication for embeddings is handled separately via
`text_hash`.

### Database

SQLite WAL mode at `~/.lss/lss.db`. Three main structures:

- `files` -- one row per indexed file (keyed by path-based `file_uid`)
- `fts` -- FTS5 virtual table for BM25 text search
- `embeddings` -- cached embedding vectors (keyed by `text_hash + model + dim`)

---

## Release process

### 1. Make your changes

```bash
git checkout -b feat/your-thing
# ... code, test, iterate ...
```

### 2. Bump version

Update the version string in **one** place:

```
pyproject.toml   ->  version = "X.Y.Z"
```

`lss` reports the installed package version via package metadata; when running
from a source checkout it falls back to reading `pyproject.toml`.

Versioning convention:
- **Patch** (0.5.1 -> 0.5.2): bug fixes, minor tweaks
- **Minor** (0.5.x -> 0.6.0): new features, non-breaking changes
- **Major** (0.x -> 1.0): breaking API/CLI changes

### 3. Commit and push

```bash
git add -A
git commit -m "feat: description of change"  # or fix:, docs:, etc.
git push origin feat/your-thing
```

Open a PR, get it merged to `main`.

### 4. Tag and publish to PyPI

Recommended: use the GitHub Actions release workflow to create the tag from the
`pyproject.toml` version (ensures tag and PyPI match):

```bash
gh workflow run release.yml
```

Publishing happens automatically on tag push via `.github/workflows/publish.yml`.

```bash
git tag v0.5.2
git push origin v0.5.2
```

The workflow will:
1. Check out the tagged commit
2. Build the package with `python -m build`
3. Verify the wheel doesn't contain secrets or test files
4. Publish to PyPI via `pypa/gh-action-pypi-publish`

Monitor progress at: https://github.com/kortix-ai/lss/actions

### 5. Verify

```bash
# Wait a minute for PyPI to propagate, then:
pip install --upgrade local-semantic-search
lss --version
```

Users can also upgrade via:

```bash
lss update                              # built-in upgrade command
```

---

## Quick reference

| Task | Command |
|------|---------|
| Run tests | `pytest` |
| Run specific test | `pytest tests/test_lss_store.py::test_name -v` |
| Lint | `ruff check .` |
| Format | `black .` |
| Type check | `mypy .` |
| Search quality eval | `lss eval` |
| Bump version | Edit `lss_cli.py` + `pyproject.toml` |
| Release to PyPI | `git tag vX.Y.Z && git push origin vX.Y.Z` |
| Build single binary | `pip install ".[build]" && python -m nuitka --onefile lss_cli.py` |
| Build Linux binary | `bash build-linux.sh` (or see `BUILD_LINUX.md`) |
