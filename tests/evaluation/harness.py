"""Search quality evaluation harness using ranx.

Orchestrates corpus generation, indexing, three-way search (BM25 / embedding / hybrid),
and metric computation against hand-labeled golden set queries.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from ranx import Qrels, Run, evaluate, compare


METRICS = [
    "ndcg@5", "ndcg@10",
    "mrr@10",
    "recall@5", "recall@10",
    "precision@5", "precision@10",
    "map@10",
]


@dataclass
class EvalResult:
    """Metrics for a single retrieval method."""
    method: str
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    mrr_10: float = 0.0
    recall_5: float = 0.0
    recall_10: float = 0.0
    precision_5: float = 0.0
    precision_10: float = 0.0
    map_10: float = 0.0
    queries_evaluated: int = 0
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    """Full evaluation report comparing all methods."""
    bm25: EvalResult = field(default_factory=lambda: EvalResult("bm25"))
    embedding: EvalResult = field(default_factory=lambda: EvalResult("embedding"))
    hybrid: EvalResult = field(default_factory=lambda: EvalResult("hybrid"))
    corpus_files: int = 0
    total_queries: int = 0

    def summary_table(self) -> str:
        """Return a formatted ASCII comparison table."""
        header = f"{'Method':<12} {'NDCG@5':>8} {'NDCG@10':>8} {'MRR@10':>8} {'Recall@5':>9} {'Recall@10':>10} {'P@5':>6} {'P@10':>6} {'MAP@10':>8} {'ms':>8}"
        sep = "-" * len(header)
        rows = []
        for r in [self.bm25, self.embedding, self.hybrid]:
            rows.append(
                f"{r.method:<12} {r.ndcg_5:>8.3f} {r.ndcg_10:>8.3f} {r.mrr_10:>8.3f} "
                f"{r.recall_5:>9.3f} {r.recall_10:>10.3f} {r.precision_5:>6.3f} "
                f"{r.precision_10:>6.3f} {r.map_10:>8.3f} {r.elapsed_ms:>8.0f}"
            )
        return f"\n{sep}\n{header}\n{sep}\n" + "\n".join(rows) + f"\n{sep}\n"

    def to_dict(self) -> dict:
        return {
            "corpus_files": self.corpus_files,
            "total_queries": self.total_queries,
            "bm25": asdict(self.bm25),
            "embedding": asdict(self.embedding),
            "hybrid": asdict(self.hybrid),
        }


class SearchEvalHarness:
    """Evaluation harness for comparing BM25, embedding, and hybrid search."""

    def __init__(self, golden_set_path: str | Path):
        with open(golden_set_path) as f:
            self.golden_set = json.load(f)
        self.queries = self.golden_set["queries"]

    def build_qrels(self, corpus_path: Path) -> Qrels:
        """Convert golden set judgments to ranx Qrels.

        Judgment keys in the golden set are relative paths (e.g. "src/auth/jwt_handler.py").
        lss indexes absolute paths and chunk IDs look like "<file_uid>::0".
        We need to map relative paths -> chunk IDs that lss actually returns.

        Since we can't predict chunk IDs upfront, we use the relative file path
        as the document ID in qrels, and we'll map search results to relative
        paths before evaluation.
        """
        qrels_dict = {}
        for q in self.queries:
            qrels_dict[q["id"]] = {
                doc_path: int(relevance)
                for doc_path, relevance in q["judgments"].items()
            }
        return Qrels.from_dict(qrels_dict)

    def run_search(self, corpus_path: Path, mode: str) -> tuple[Run, float]:
        """Run search for all queries and return a ranx Run + elapsed time.

        The Run maps query_id -> {relative_file_path: score}.
        We aggregate chunk-level scores to file-level by taking the max score
        per file, since our golden set judges at file granularity.
        """
        if mode == "bm25":
            from semantic_search import search_bm25_only as search_fn
        elif mode == "embedding":
            from semantic_search import search_embeddings_only as search_fn
        elif mode == "hybrid":
            from semantic_search import search_hybrid as search_fn
        else:
            raise ValueError(f"Unknown mode: {mode}")

        results = {}
        t0 = time.time()

        for q in self.queries:
            query_text = q["text"]
            # search_fn returns a list (one dict per query), we pass single query
            raw = search_fn(str(corpus_path), [query_text])
            if not raw or not raw[0]:
                results[q["id"]] = {}
                continue

            # raw[0] is {chunk_id: score} — aggregate to file-level
            chunk_scores = raw[0]
            file_scores = self._aggregate_to_files(chunk_scores, corpus_path)
            results[q["id"]] = file_scores

        elapsed_ms = (time.time() - t0) * 1000
        run = Run.from_dict(results, name=mode)
        return run, elapsed_ms

    def _aggregate_to_files(self, chunk_scores: dict, corpus_path: Path) -> dict:
        """Aggregate chunk_id scores to relative file paths.

        chunk_id format from lss: "<file_uid>::<chunk_index>"
        We need to resolve file_uid -> absolute path -> relative path.

        Simpler approach: chunk IDs from _search_components use the FTS id field
        which is "<file_uid>::<chunk_idx>". We query the DB to resolve file_uid
        to file_path, then make it relative to corpus_path.
        """
        import lss_store

        if not chunk_scores:
            return {}

        # Get file_uid -> path mapping
        file_scores = {}
        con = lss_store._init_db()

        for chunk_id, score in chunk_scores.items():
            # Extract file_uid from chunk_id
            if "::" in chunk_id:
                file_uid = chunk_id.split("::")[0]
            else:
                file_uid = chunk_id

            # Look up file path
            row = con.execute(
                "SELECT path FROM files WHERE file_uid = ?", (file_uid,)
            ).fetchone()
            if row:
                abs_path = row[0]
                try:
                    rel_path = str(Path(abs_path).relative_to(corpus_path))
                except ValueError:
                    rel_path = abs_path

                # Max score per file (file may have multiple chunks)
                if rel_path not in file_scores or score > file_scores[rel_path]:
                    file_scores[rel_path] = float(score)

        con.close()
        return file_scores

    def evaluate_run(self, run: Run, qrels: Qrels, elapsed_ms: float) -> EvalResult:
        """Evaluate a single Run against qrels."""
        scores = evaluate(qrels, run, METRICS, return_mean=True)

        return EvalResult(
            method=run.name,
            ndcg_5=scores["ndcg@5"],
            ndcg_10=scores["ndcg@10"],
            mrr_10=scores["mrr@10"],
            recall_5=scores["recall@5"],
            recall_10=scores["recall@10"],
            precision_5=scores["precision@5"],
            precision_10=scores["precision@10"],
            map_10=scores["map@10"],
            queries_evaluated=len(self.queries),
            elapsed_ms=elapsed_ms,
        )

    def full_evaluation(self, corpus_path: Path) -> EvalReport:
        """Run the complete three-way evaluation.

        Returns an EvalReport with BM25, embedding, and hybrid results.
        """
        from evaluation.corpus import CORPUS_FILES

        qrels = self.build_qrels(corpus_path)
        report = EvalReport(
            corpus_files=len(CORPUS_FILES),
            total_queries=len(self.queries),
        )

        # Run all three modes
        for mode in ("bm25", "embedding", "hybrid"):
            run, elapsed = self.run_search(corpus_path, mode)
            result = self.evaluate_run(run, qrels, elapsed)
            setattr(report, mode, result)

        return report

    def compare_methods(self, corpus_path: Path) -> str:
        """Run all methods and return a ranx statistical comparison table."""
        qrels = self.build_qrels(corpus_path)
        runs = []
        for mode in ("bm25", "embedding", "hybrid"):
            run, _ = self.run_search(corpus_path, mode)
            runs.append(run)

        return compare(
            qrels=qrels,
            runs=runs,
            metrics=METRICS[:4],  # Top metrics only for readability
            max_p=0.05,
        )
