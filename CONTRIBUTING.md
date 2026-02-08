# Contributing to lss

Thanks for helping make lss better.

## Workflow
1. Fork -> branch: `git checkout -b feat/your-thing`
2. Commit: conventional-ish messages (`feat:`, `fix:`, etc.)
3. Push & open a PR
4. Keep PRs scoped to one change; add tests/docs as needed

## Dev Setup

```bash
# create venv & install deps (incl. linters/tests)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# or with uv:
uv venv && uv sync -e dev
```

## Running

```bash
# CLI
lss "authentication JWT"
lss "deploy kubernetes" ~/Projects
lss status
lss eval

# or without installing:
python -m lss_cli "authentication JWT"
```

Programmatic:

```python
from semantic_search import semantic_search
from lss_store import discover_files, ingest_many

files = discover_files("/path/to/project")
ingest_many(files)
results = semantic_search("/path/to/project", ["reset 2FA"])
print(results[0][:3])
```

> Put your `.env` in CWD / `$LSS_DIR` / `~/.config/lss` / package dir. Need `OPENAI_API_KEY`.

## Quality Gate

```bash
ruff check .
black --check .
mypy .
pytest                    # 90 tests, ~2 min (uses OpenAI API)
pytest -k "not beir"      # skip BEIR benchmarks (~10 min)
lss eval                  # search quality evaluation
```

## Style & Guidelines

* Python >= 3.9; line length=100; no heavy deps
* Follow existing patterns; prefer small functions & pure helpers
* Add/adjust tests for new behavior
* Update docs if CLI/API changes

## Versioning/Caches

* `VERSION_KEY` is auto-derived: `{model}:{dim}:p{PREPROC_VER}:c{CHUNKER_VER}`
* If you change preprocessing or chunking logic, bump `PREPROC_VER`/`CHUNKER_VER` constants

## Reporting Issues

Include:

* Repro steps, expected vs. actual
* OS, Python version, `lss --version`
* Relevant logs/output (redact keys)
* Minimal file/query example when possible

## Optional: Single-Binary (maintainers)

```bash
pip install ".[build]"
python -m nuitka \
  --onefile \
  --standalone \
  --static-libpython=no \
  --follow-imports \
  --enable-plugin=numpy \
  --output-filename=lss \
  lss_cli.py

# then: ./lss "query"
```
