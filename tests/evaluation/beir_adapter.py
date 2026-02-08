"""Adapter to run BEIR benchmark datasets through lss.

Loads a BEIR dataset via ir-datasets, writes documents as text files
to a temp directory, indexes them with lss, then runs search queries
and maps results back to BEIR doc IDs for evaluation with ranx.
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass

import ir_datasets
from ranx import Qrels, Run, evaluate

# Published BEIR baselines (NDCG@10) from:
# - BEIR paper (Thakur et al., NeurIPS 2021)
# - MTEB leaderboard
# - Elastic blog posts
PUBLISHED_BASELINES = {
    "beir/scifact/test": {
        "BM25 (Anserini)": 0.665,
        "BM25 (BEIR official)": 0.665,
        "ANCE (dense)": 0.507,
        "DPR": 0.318,
        "TAS-B": 0.643,
        "ColBERTv2": 0.693,
        "GTR-XXL": 0.662,
        "text-embedding-3-small": 0.694,  # OpenAI MTEB
        "text-embedding-3-large": 0.735,  # OpenAI MTEB
        "Cohere embed-v3": 0.717,
        "Voyage-2": 0.713,
    },
    "beir/nfcorpus/test": {
        "BM25 (Anserini)": 0.325,
        "BM25 (BEIR official)": 0.325,
        "ANCE (dense)": 0.237,
        "DPR": 0.189,
        "TAS-B": 0.319,
        "ColBERTv2": 0.338,
        "text-embedding-3-small": 0.336,
        "text-embedding-3-large": 0.361,
        "Cohere embed-v3": 0.350,
    },
}

METRICS = [
    "ndcg@10", "ndcg@5",
    "mrr@10",
    "recall@10", "recall@100",
    "precision@10",
    "map@10",
]


@dataclass
class BeirResult:
    """Results from running a BEIR benchmark."""
    dataset: str
    method: str
    ndcg_10: float
    ndcg_5: float
    mrr_10: float
    recall_10: float
    recall_100: float
    precision_10: float
    map_10: float
    num_queries: int
    num_docs: int
    elapsed_s: float


def load_beir_dataset(dataset_id: str):
    """Load a BEIR dataset via ir-datasets.

    Returns (corpus, queries, qrels) where:
    - corpus: dict of {doc_id: {"title": str, "text": str}}
    - queries: dict of {query_id: str}
    - qrels: dict of {query_id: {doc_id: int}}
    """
    ds = ir_datasets.load(dataset_id)

    corpus = {}
    for doc in ds.docs_iter():
        corpus[doc.doc_id] = {
            "title": getattr(doc, "title", ""),
            "text": getattr(doc, "text", ""),
        }

    queries = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text

    qrels = {}
    for qr in ds.qrels_iter():
        if qr.query_id not in qrels:
            qrels[qr.query_id] = {}
        qrels[qr.query_id][qr.doc_id] = qr.relevance

    return corpus, queries, qrels


def write_corpus_to_disk(corpus: dict, base_path: Path) -> dict:
    """Write BEIR corpus as text files. Returns {doc_id: file_path} mapping.

    Each document becomes a .txt file named by doc_id.
    Content format: "Title\\n\\nBody text"
    Files are organized in subdirectories of 500 to avoid filesystem issues.
    """
    base_path.mkdir(parents=True, exist_ok=True)
    mapping = {}

    for i, (doc_id, doc) in enumerate(corpus.items()):
        # Organize into subdirs of 500 files
        subdir = base_path / f"shard_{i // 500:03d}"
        subdir.mkdir(exist_ok=True)

        # Sanitize doc_id for filename
        safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
        filepath = subdir / f"{safe_id}.txt"

        content = ""
        if doc["title"]:
            content = doc["title"] + "\n\n"
        content += doc["text"]

        filepath.write_text(content, encoding="utf-8")
        mapping[doc_id] = filepath

    return mapping


def index_beir_corpus(corpus_path: Path) -> float:
    """Index the written corpus with lss. Returns elapsed time in seconds."""
    from lss_store import discover_files, ingest_many

    t0 = time.time()
    all_paths, new_paths, already = discover_files(corpus_path)
    if new_paths:
        ingest_many(new_paths)
    return time.time() - t0


def _resolve_file_to_docid(filepath: str, corpus_path: Path) -> str:
    """Convert a file path back to a BEIR doc ID.

    Files are named: shard_XXX/<doc_id>.txt
    Resolves symlinks (macOS /tmp -> /private/tmp) for reliable matching.
    """
    try:
        resolved_file = Path(filepath).resolve()
        resolved_corpus = corpus_path.resolve()
        rel = resolved_file.relative_to(resolved_corpus)
        # Remove shard prefix and .txt extension
        return rel.stem
    except (ValueError, Exception):
        # Fallback: just extract the filename stem
        return Path(filepath).stem


def run_beir_search(corpus_path: Path, queries: dict, mode: str,
                    doc_mapping: dict = None) -> tuple[Run, float]:
    """Run lss search for all BEIR queries.

    Returns (ranx Run, elapsed_seconds).
    The Run maps query_id -> {doc_id: score}.
    """
    if mode == "bm25":
        from semantic_search import search_bm25_only as search_fn
    elif mode == "embedding":
        from semantic_search import search_embeddings_only as search_fn
    elif mode == "hybrid":
        from semantic_search import search_hybrid as search_fn
    else:
        raise ValueError(f"Unknown mode: {mode}")

    import lss_store

    results = {}
    t0 = time.time()

    # Resolve corpus_path once (handles macOS /tmp -> /private/tmp symlinks)
    resolved_corpus = corpus_path.resolve()

    con = lss_store._init_db()

    query_items = list(queries.items())
    for i, (qid, query_text) in enumerate(query_items):
        raw = search_fn(str(corpus_path), [query_text])
        if not raw or not raw[0]:
            results[qid] = {}
            continue

        # Map chunk scores back to BEIR doc IDs
        doc_scores = {}
        for chunk_id, score in raw[0].items():
            file_uid = chunk_id.split("::")[0] if "::" in chunk_id else chunk_id
            row = con.execute(
                "SELECT path FROM files WHERE file_uid = ?", (file_uid,)
            ).fetchone()
            if row:
                doc_id = _resolve_file_to_docid(row[0], resolved_corpus)
                # Max score per doc (doc may have multiple chunks)
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = float(score)

        results[qid] = doc_scores

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{mode}] {i+1}/{len(query_items)} queries ({elapsed:.1f}s)")

    con.close()
    elapsed = time.time() - t0
    run = Run.from_dict(results, name=mode)
    return run, elapsed


def evaluate_beir(dataset_id: str, corpus_path: Path, queries: dict,
                  qrels_dict: dict, mode: str,
                  doc_mapping: dict = None) -> BeirResult:
    """Run search + evaluation for one mode on a BEIR dataset."""

    qrels = Qrels.from_dict(qrels_dict)
    run, elapsed = run_beir_search(corpus_path, queries, mode, doc_mapping)
    scores = evaluate(qrels, run, METRICS, return_mean=True)

    # Get corpus size from disk
    num_docs = sum(1 for _ in corpus_path.rglob("*.txt"))

    return BeirResult(
        dataset=dataset_id,
        method=mode,
        ndcg_10=scores["ndcg@10"],
        ndcg_5=scores["ndcg@5"],
        mrr_10=scores["mrr@10"],
        recall_10=scores["recall@10"],
        recall_100=scores["recall@100"],
        precision_10=scores["precision@10"],
        map_10=scores["map@10"],
        num_queries=len(queries),
        num_docs=num_docs,
        elapsed_s=elapsed,
    )


def format_comparison_table(results: list[BeirResult],
                            dataset_id: str) -> str:
    """Format results with published baselines for comparison."""

    baselines = PUBLISHED_BASELINES.get(dataset_id, {})

    lines = []
    lines.append("")
    header = f"{'System':<35} {'NDCG@10':>8} {'MRR@10':>8} {'R@10':>8} {'R@100':>8}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # Our results
    for r in results:
        label = f"lss {r.method}"
        lines.append(
            f"  {label:<33} {r.ndcg_10:>8.3f} {r.mrr_10:>8.3f} "
            f"{r.recall_10:>8.3f} {r.recall_100:>8.3f}"
        )

    lines.append(sep)

    # Published baselines
    for name, ndcg in sorted(baselines.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:<33} {ndcg:>8.3f}")

    lines.append(sep)
    return "\n".join(lines)
