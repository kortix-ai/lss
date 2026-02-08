"""Search quality evaluation tests.

Generates a realistic project corpus, indexes it, runs BM25 / embedding / hybrid
search against 40 hand-labeled queries, and evaluates with standard IR metrics
(NDCG, MRR, Recall, Precision, MAP) via ranx.

These tests require OPENAI_API_KEY since embedding search needs the API.
They are gated behind the `require_openai` fixture.

Run with:
    pytest tests/test_search_quality.py -v -s
"""

import os
import sys
import json
import time
from pathlib import Path

import pytest

# Ensure evaluation package is importable
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.corpus import generate_corpus, CORPUS_FILES
from evaluation.harness import SearchEvalHarness, EvalReport

from lss_store import discover_files, ingest_many, _file_cache
from lss_cli import _C

# ── Fixtures ─────────────────────────────────────────────────────────────────

_skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)


GOLDEN_SET_PATH = Path(__file__).parent / "evaluation" / "golden_set.json"


def _setup_lss_env(lss_dir):
    """Point all lss modules at a given LSS_DIR."""
    import lss_config
    import lss_store
    import semantic_search

    os.environ["LSS_DIR"] = str(lss_dir)
    os.environ["LSS_ENV"] = "test"
    os.environ["OAI_TIMEOUT"] = "15"

    lss_config.LSS_DIR = lss_dir
    lss_config.LSS_DB = lss_dir / "lss.db"
    lss_config.CONFIG_FILE = lss_dir / "config.json"
    lss_store.LSS_DIR = lss_dir
    lss_store.LSS_DB = lss_dir / "lss.db"
    lss_store._file_cache.clear()
    semantic_search.OAI_TIMEOUT = 15.0


# Module-level state: generated once, reused across tests
_eval_lss_dir = None
_eval_corpus_path = None


@pytest.fixture
def eval_corpus(tmp_path):
    """Generate the test corpus and index it.

    Uses module-level caching: the first call generates and indexes the corpus,
    subsequent calls just re-point the lss modules at the same LSS_DIR.
    """
    global _eval_lss_dir, _eval_corpus_path

    if _eval_corpus_path is None or not _eval_corpus_path.exists():
        # First call — generate corpus and index
        base = tmp_path
        corpus_path = base / "project"
        lss_dir = base / ".lss"
        lss_dir.mkdir(exist_ok=True)

        _setup_lss_env(lss_dir)

        files = generate_corpus(corpus_path)
        all_paths, new_paths, already = discover_files(corpus_path)
        ingest_many(new_paths)

        _eval_lss_dir = lss_dir
        _eval_corpus_path = corpus_path
    else:
        # Subsequent calls — just re-point modules at the existing DB
        _setup_lss_env(_eval_lss_dir)

    return _eval_corpus_path


@pytest.fixture
def harness():
    return SearchEvalHarness(GOLDEN_SET_PATH)


# ── BM25-only tests (no OpenAI needed) ──────────────────────────────────────


class TestBM25Quality:
    """BM25 search quality — no API calls required."""

    def test_bm25_returns_results(self, eval_corpus, harness):
        """BM25 should return results for most queries."""
        from semantic_search import search_bm25_only

        empty_count = 0
        for q in harness.queries:
            results = search_bm25_only(str(eval_corpus), [q["text"]])
            if not results[0]:
                empty_count += 1

        # Allow at most 5 queries to return empty (some semantic queries may not match BM25)
        assert empty_count <= 5, f"BM25 returned empty for {empty_count}/{len(harness.queries)} queries"

    def test_bm25_keyword_queries_find_target(self, eval_corpus, harness):
        """BM25 should find the primary relevant doc for keyword queries."""
        from semantic_search import search_bm25_only

        keyword_queries = [q for q in harness.queries if q["category"] == "keyword"]
        hits = 0

        for q in keyword_queries:
            results = search_bm25_only(str(eval_corpus), [q["text"]])
            if not results[0]:
                continue

            # Get the most relevant doc from judgments
            top_relevant = [doc for doc, rel in q["judgments"].items() if rel == 2]
            # Check if any top-relevant doc appears in results (via file path)
            found_chunks = results[0]
            found_files = harness._aggregate_to_files(found_chunks, eval_corpus)

            if any(doc in found_files for doc in top_relevant):
                hits += 1

        hit_rate = hits / len(keyword_queries) if keyword_queries else 0
        assert hit_rate >= 0.5, f"BM25 keyword hit rate {hit_rate:.1%} < 50%"
        print(f"\n  [EVAL] BM25 keyword hit rate: {hit_rate:.1%} ({hits}/{len(keyword_queries)})")


# ── Full three-way evaluation (requires OpenAI) ─────────────────────────────


@_skip_no_openai
class TestSearchQuality:
    """Full three-way search quality evaluation."""

    def test_full_evaluation(self, eval_corpus, harness):
        """Run the complete evaluation and print the report."""
        report = harness.full_evaluation(eval_corpus)
        print(report.summary_table())

        # Store for other tests to access
        self.__class__._report = report

    def test_hybrid_ndcg_above_minimum(self, eval_corpus, harness):
        """Hybrid search NDCG@10 should meet a minimum quality bar.

        Baseline measurement: 0.920 (2026-02-08).
        Threshold set at 0.70 to allow for OpenAI model changes and
        embedding variance while catching real regressions.
        """
        report = harness.full_evaluation(eval_corpus)

        assert report.hybrid.ndcg_10 > 0.70, (
            f"Hybrid NDCG@10={report.hybrid.ndcg_10:.3f} is below minimum 0.70"
        )
        print(f"\n  [EVAL] Hybrid NDCG@10: {report.hybrid.ndcg_10:.3f}")

    def test_hybrid_mrr_above_minimum(self, eval_corpus, harness):
        """Hybrid MRR@10 should meet a minimum quality bar.

        Baseline measurement: 1.000 (2026-02-08).
        Threshold set at 0.80 — the first result should usually be relevant.
        """
        report = harness.full_evaluation(eval_corpus)

        assert report.hybrid.mrr_10 > 0.80, (
            f"Hybrid MRR@10={report.hybrid.mrr_10:.3f} is below minimum 0.80"
        )
        print(f"\n  [EVAL] Hybrid MRR@10: {report.hybrid.mrr_10:.3f}")

    def test_hybrid_beats_or_matches_bm25(self, eval_corpus, harness):
        """Hybrid search should be at least as good as BM25-only on NDCG@10."""
        report = harness.full_evaluation(eval_corpus)

        # Allow a small margin — hybrid could be slightly worse on some queries
        margin = 0.02
        assert report.hybrid.ndcg_10 >= report.bm25.ndcg_10 - margin, (
            f"Hybrid NDCG@10={report.hybrid.ndcg_10:.3f} significantly worse than "
            f"BM25 NDCG@10={report.bm25.ndcg_10:.3f}"
        )

    def test_hybrid_beats_or_matches_embedding(self, eval_corpus, harness):
        """Hybrid search should be at least as good as embedding-only on NDCG@10."""
        report = harness.full_evaluation(eval_corpus)

        margin = 0.02
        assert report.hybrid.ndcg_10 >= report.embedding.ndcg_10 - margin, (
            f"Hybrid NDCG@10={report.hybrid.ndcg_10:.3f} significantly worse than "
            f"Embedding NDCG@10={report.embedding.ndcg_10:.3f}"
        )

    def test_per_category_quality(self, eval_corpus, harness):
        """Check quality across different query categories."""
        from semantic_search import search_hybrid
        from ranx import Qrels, Run, evaluate

        categories = {}
        for q in harness.queries:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = {"queries": [], "qrels": {}, "results": {}}
            categories[cat]["queries"].append(q)

        print("\n  [EVAL] Per-category NDCG@10:")
        for cat, data in sorted(categories.items()):
            qrels_dict = {}
            results_dict = {}

            for q in data["queries"]:
                qrels_dict[q["id"]] = {
                    doc: int(rel) for doc, rel in q["judgments"].items()
                }
                raw = search_hybrid(str(eval_corpus), [q["text"]])
                if raw and raw[0]:
                    results_dict[q["id"]] = harness._aggregate_to_files(raw[0], eval_corpus)
                else:
                    results_dict[q["id"]] = {}

            if not results_dict:
                continue

            qrels = Qrels.from_dict(qrels_dict)
            run = Run.from_dict(results_dict, name=cat)
            scores = evaluate(qrels, run, ["ndcg@10", "mrr@10"], return_mean=True)
            print(f"    {cat:<16} NDCG@10={scores['ndcg@10']:.3f}  MRR@10={scores['mrr@10']:.3f}  ({len(data['queries'])} queries)")


# ── Regression guard (run independently) ─────────────────────────────────────


@_skip_no_openai
class TestSearchRegression:
    """Regression tests to catch quality degradation.

    These tests use conservative thresholds. After the initial run,
    update the thresholds to actual measured values minus a margin.
    """

    def test_no_empty_hybrid_results(self, eval_corpus, harness):
        """Hybrid search should return results for almost all queries."""
        from semantic_search import search_hybrid

        empty = 0
        for q in harness.queries:
            results = search_hybrid(str(eval_corpus), [q["text"]])
            if not results[0]:
                empty += 1

        max_empty = 3  # At most 3 queries can return empty
        assert empty <= max_empty, f"Hybrid returned empty for {empty} queries (max={max_empty})"
        print(f"\n  [EVAL] Hybrid empty results: {empty}/{len(harness.queries)}")

    def test_recall_at_10_above_minimum(self, eval_corpus, harness):
        """Recall@10 should be reasonable — we should find most relevant docs.

        Baseline measurement: 0.943 (2026-02-08).
        Threshold set at 0.70.
        """
        report = harness.full_evaluation(eval_corpus)
        assert report.hybrid.recall_10 > 0.70, (
            f"Hybrid Recall@10={report.hybrid.recall_10:.3f} is below minimum 0.70"
        )
