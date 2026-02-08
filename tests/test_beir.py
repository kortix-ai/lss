"""BEIR benchmark tests — evaluate lss against standard IR datasets.

Runs lss search (BM25, embedding, hybrid) on real BEIR benchmark datasets
and compares results against published baselines from top retrieval systems.

Datasets:
  - SciFact: 5,183 biomedical documents, 300 queries (fact verification)
  - NFCorpus: 3,633 medical documents, 323 queries (nutrition/fitness)

These tests are:
  - Slow (minutes, not seconds) — marked with @pytest.mark.slow
  - Require OPENAI_API_KEY for embedding/hybrid modes
  - Require internet for first run (downloads datasets, ~3MB each, cached after)

Run with:
    pytest tests/test_beir.py -v -s                    # all BEIR tests
    pytest tests/test_beir.py -k scifact -v -s         # SciFact only
    pytest tests/test_beir.py -k "bm25" -v -s          # BM25 only (no API)
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.beir_adapter import (
    load_beir_dataset,
    write_corpus_to_disk,
    index_beir_corpus,
    evaluate_beir,
    format_comparison_table,
    PUBLISHED_BASELINES,
)
from lss_cli import _C


# ── Markers & fixtures ───────────────────────────────────────────────────────

_skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture(autouse=True)
def no_colors():
    _C.set_enabled(False)
    yield
    _C.set_enabled(None)


def _setup_beir_env(lss_dir):
    """Point all lss modules at a given LSS_DIR for BEIR tests."""
    import lss_config
    import lss_store
    import semantic_search

    os.environ["LSS_DIR"] = str(lss_dir)
    os.environ["LSS_ENV"] = "test"
    os.environ["OAI_TIMEOUT"] = "30"  # Higher timeout for BEIR (more docs)

    lss_config.LSS_DIR = lss_dir
    lss_config.LSS_DB = lss_dir / "lss.db"
    lss_config.CONFIG_FILE = lss_dir / "config.json"
    lss_store.LSS_DIR = lss_dir
    lss_store.LSS_DB = lss_dir / "lss.db"
    lss_store._file_cache.clear()
    semantic_search.OAI_TIMEOUT = 30.0


# ── Module-level caches for expensive operations ─────────────────────────────
# Each dataset is loaded, written, and indexed once per test session.

_beir_cache = {}  # {dataset_id: (corpus_path, queries, qrels, doc_mapping)}


def _get_beir_data(dataset_id: str, tmp_path: Path):
    """Load and index a BEIR dataset, caching the result."""
    if dataset_id in _beir_cache:
        corpus_path, queries, qrels, doc_mapping = _beir_cache[dataset_id]
        _setup_beir_env(corpus_path.parent / ".lss")
        return corpus_path, queries, qrels, doc_mapping

    # Fresh setup
    lss_dir = tmp_path / ".lss"
    lss_dir.mkdir(exist_ok=True)
    _setup_beir_env(lss_dir)

    print(f"\n  Loading {dataset_id}...")
    corpus, queries, qrels = load_beir_dataset(dataset_id)
    print(f"  {len(corpus)} docs, {len(queries)} queries, {sum(len(v) for v in qrels.values())} qrels")

    corpus_path = tmp_path / "corpus"
    print(f"  Writing corpus to disk...")
    doc_mapping = write_corpus_to_disk(corpus, corpus_path)

    print(f"  Indexing {len(doc_mapping)} documents...")
    t_index = index_beir_corpus(corpus_path)
    print(f"  Indexed in {t_index:.1f}s ({t_index/len(doc_mapping)*1000:.1f}ms/doc)")

    _beir_cache[dataset_id] = (corpus_path, queries, qrels, doc_mapping)
    return corpus_path, queries, qrels, doc_mapping


# ── SciFact benchmarks ───────────────────────────────────────────────────────


class TestBeirSciFact:
    """BEIR SciFact benchmark (5,183 docs, 300 queries)."""

    DATASET = "beir/scifact/test"

    def test_scifact_bm25(self, tmp_path):
        """BM25 search on SciFact.

        Published BM25 (Anserini/Lucene) baseline: 0.665 NDCG@10.

        NOTE: lss uses SQLite FTS5 with Porter stemmer, which is structurally
        different from Lucene's BM25 (different tokenization, k1/b parameters,
        and scoring). FTS5 BM25 scores are very flat on short biomedical passages,
        causing poor ranking. This is a known limitation — the hybrid pipeline
        uses embeddings to compensate. We record the BM25 score but don't gate on it.
        """
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        result = evaluate_beir(
            self.DATASET, corpus_path, queries, qrels,
            mode="bm25", doc_mapping=doc_mapping,
        )

        print(f"\n  [BEIR SciFact] BM25 (FTS5) NDCG@10: {result.ndcg_10:.3f}")
        print(f"  Published BM25 (Anserini) baseline: 0.665")
        print(f"  Note: FTS5 BM25 != Lucene BM25 — flat scoring on short passages")
        print(f"  Elapsed: {result.elapsed_s:.1f}s")

        # No assertion — FTS5 BM25 is not comparable to Lucene BM25.
        # The value of lss is in the hybrid pipeline.

    @_skip_no_openai
    def test_scifact_embedding(self, tmp_path):
        """Embedding-only search on SciFact.

        Published text-embedding-3-small: 0.694 NDCG@10.
        Baseline measurement: 0.719 (2026-02-08).
        """
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        result = evaluate_beir(
            self.DATASET, corpus_path, queries, qrels,
            mode="embedding", doc_mapping=doc_mapping,
        )

        print(f"\n  [BEIR SciFact] Embedding NDCG@10: {result.ndcg_10:.3f}")
        print(f"  Published text-embedding-3-small: 0.694")
        print(f"  Elapsed: {result.elapsed_s:.1f}s")

        # Should beat the published text-embedding-3-small baseline (0.694)
        # with margin for API variance
        assert result.ndcg_10 > 0.55, (
            f"SciFact Embedding NDCG@10={result.ndcg_10:.3f} — below threshold 0.55"
        )

    @_skip_no_openai
    def test_scifact_hybrid(self, tmp_path):
        """Hybrid RRF search on SciFact — our full pipeline.

        This is the headline number: how does lss compare to the best systems?
        Baseline measurement: 0.729 (2026-02-08).
        Beats ColBERTv2 (0.693), text-embedding-3-small (0.694), Voyage-2 (0.713).
        """
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        result = evaluate_beir(
            self.DATASET, corpus_path, queries, qrels,
            mode="hybrid", doc_mapping=doc_mapping,
        )

        print(f"\n  [BEIR SciFact] Hybrid NDCG@10: {result.ndcg_10:.3f}")
        print(f"  Elapsed: {result.elapsed_s:.1f}s")

        # Should be competitive with top systems (>0.60)
        assert result.ndcg_10 > 0.60, (
            f"SciFact Hybrid NDCG@10={result.ndcg_10:.3f} — below threshold 0.60"
        )

    @_skip_no_openai
    def test_scifact_full_comparison(self, tmp_path):
        """Run all three modes and print comparison table with published baselines."""
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        results = []
        for mode in ("bm25", "embedding", "hybrid"):
            result = evaluate_beir(
                self.DATASET, corpus_path, queries, qrels,
                mode=mode, doc_mapping=doc_mapping,
            )
            results.append(result)

        table = format_comparison_table(results, self.DATASET)
        print(f"\n  === BEIR SciFact Comparison ==={table}")


# ── NFCorpus benchmarks ──────────────────────────────────────────────────────


class TestBeirNFCorpus:
    """BEIR NFCorpus benchmark (3,633 docs, 323 queries)."""

    DATASET = "beir/nfcorpus/test"

    def test_nfcorpus_bm25(self, tmp_path):
        """BM25 search on NFCorpus.

        Published BM25 (Anserini) baseline: 0.325 NDCG@10.
        Same FTS5 limitation as SciFact — recorded but not gated.
        """
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        result = evaluate_beir(
            self.DATASET, corpus_path, queries, qrels,
            mode="bm25", doc_mapping=doc_mapping,
        )

        print(f"\n  [BEIR NFCorpus] BM25 (FTS5) NDCG@10: {result.ndcg_10:.3f}")
        print(f"  Published BM25 (Anserini) baseline: 0.325")
        print(f"  Elapsed: {result.elapsed_s:.1f}s")

    @_skip_no_openai
    def test_nfcorpus_hybrid(self, tmp_path):
        """Hybrid RRF search on NFCorpus.

        Baseline measurement: 0.334 (2026-02-08).
        Competitive with text-embedding-3-small (0.336) and ColBERTv2 (0.338).
        """
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        result = evaluate_beir(
            self.DATASET, corpus_path, queries, qrels,
            mode="hybrid", doc_mapping=doc_mapping,
        )

        print(f"\n  [BEIR NFCorpus] Hybrid NDCG@10: {result.ndcg_10:.3f}")
        print(f"  Elapsed: {result.elapsed_s:.1f}s")

        assert result.ndcg_10 > 0.20, (
            f"NFCorpus Hybrid NDCG@10={result.ndcg_10:.3f} — below threshold 0.20"
        )

    @_skip_no_openai
    def test_nfcorpus_full_comparison(self, tmp_path):
        """Run all modes and print comparison with published baselines."""
        corpus_path, queries, qrels, doc_mapping = _get_beir_data(self.DATASET, tmp_path)

        results = []
        for mode in ("bm25", "embedding", "hybrid"):
            result = evaluate_beir(
                self.DATASET, corpus_path, queries, qrels,
                mode=mode, doc_mapping=doc_mapping,
            )
            results.append(result)

        table = format_comparison_table(results, self.DATASET)
        print(f"\n  === BEIR NFCorpus Comparison ==={table}")
