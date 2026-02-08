# LSS Search Quality Evaluation

> Version 0.4.0 — February 2026

LSS combines BM25 keyword search with OpenAI embeddings via Reciprocal Rank Fusion. This document reports search quality results on both internal and standard IR benchmarks, compared against published baselines.

---

## Methods

| Method | Description |
|--------|-------------|
| **BM25** | Keyword matching via SQLite FTS5 with custom BM25 re-scoring (k1=1.2, b=0.75). Uses corpus statistics from `fts_vocab` for proper TF saturation and IDF weighting. |
| **Embedding** | Semantic similarity via OpenAI `text-embedding-3-small` (256 dims). Cosine similarity between query and document vectors. |
| **Hybrid** | Both BM25 and embedding results merged via Reciprocal Rank Fusion (k=60), with post-fusion Jaccard/phrase boosts and MMR diversity re-ranking. **This is the default `lss` search mode.** |

## Metrics

| Metric | Meaning |
|--------|---------|
| **NDCG@10** | Normalized Discounted Cumulative Gain at rank 10. Measures ranking quality — rewards placing highly relevant results at the top. **Primary metric in IR research.** |
| **MRR@10** | Mean Reciprocal Rank. How quickly the first relevant result appears. 1.0 = always at position 1. |
| **Recall@10** | Fraction of all relevant documents found in the top 10 results. |
| **P@10** | Precision at 10 — fraction of top-10 results that are relevant. |
| **MAP@10** | Mean Average Precision — average precision across all recall levels. |

---

## 1. Golden Set (Internal Benchmark)

**Corpus:** 30-file synthetic software project (auth, API, DB, deploy, monitoring, security, config, docs, tests, scripts).
**Queries:** 40 hand-labeled queries with 3-level graded relevance (0 = not relevant, 1 = mentions topic, 2 = directly answers).
**Categories:** keyword (22), conceptual (9), procedural (5), multi-concept (2), short/vague (2).

### Results

```
Method         NDCG@5  NDCG@10   MRR@10  Recall@5  Recall@10    P@5   P@10   MAP@10
─────────────────────────────────────────────────────────────────────────────────────
bm25            0.868    0.888    0.971     0.833      0.895  0.470  0.255    0.805
embedding       0.860    0.901    0.988     0.801      0.930  0.450  0.267    0.796
hybrid          0.911    0.932    1.000     0.886      0.948  0.500  0.273    0.857
```

### Key findings

- **Hybrid wins across every metric.** NDCG@10 of 0.932, perfect MRR@10 of 1.000.
- **BM25 matches embedding** on NDCG@10 (0.888 vs 0.901) thanks to custom re-scoring. Before v0.4.0, BM25 NDCG@10 was 0.204 — a **4.4x improvement**.
- **BM25 keyword hit rate: 100%** (22/22 keyword queries return the correct file at rank 1).
- **Zero empty results** — all 40 queries return relevant documents.

### Per-category breakdown

```
Category         Queries   NDCG@10   MRR@10
────────────────────────────────────────────
keyword              22     0.951     1.000
procedural            5     0.984     1.000
conceptual            9     0.921     1.000
multi_concept         2     0.901     1.000
short_vague           2     0.676     1.000
```

Short/vague queries ("monitoring", "security") are hardest — single-word queries with many relevant files spread across the corpus. Even so, MRR is perfect (first result is always correct).

### Reproduce

```bash
lss eval           # runs golden set, prints table
lss eval --json    # machine-readable output
```

---

## 2. BEIR SciFact

**Dataset:** 5,183 biomedical abstracts, 300 fact-checking queries. Standard benchmark from [BEIR](https://github.com/beir-cellar/beir).
**Protocol:** Full corpus indexed into lss. Each query searched with `search_bm25_only()`, `search_embeddings_only()`, and `search_hybrid()`. Scores computed via [ranx](https://github.com/AmenRa/ranx).

### NDCG@10 Comparison

```
System                                NDCG@10
────────────────────────────────────────────────
text-embedding-3-large  (OpenAI)       0.735
lss hybrid                             0.729
Cohere embed-v3                        0.717
lss embedding-only                     0.719
Voyage-2                               0.713
text-embedding-3-small  (OpenAI)       0.694
ColBERTv2                              0.693
BM25 (Anserini/Lucene)                 0.665
```

**lss hybrid outperforms** ColBERTv2, Voyage-2, Cohere embed-v3, and text-embedding-3-small. It is within 0.006 of text-embedding-3-large (a 3072-dim model, 12x larger vectors than lss's 256-dim).

### What this means

- lss uses `text-embedding-3-small` at 256 dimensions — the cheapest, smallest OpenAI embedding.
- By fusing BM25 with embeddings, lss matches or beats systems using much larger/more expensive models.
- ColBERTv2 requires a dedicated GPU for inference. lss runs on a laptop with an API call.

---

## 3. BEIR NFCorpus

**Dataset:** 3,633 medical documents (mix of titles, abstracts, full articles), 323 queries. Considered harder than SciFact due to vocabulary complexity.

### NDCG@10 Comparison

```
System                                NDCG@10
────────────────────────────────────────────────
text-embedding-3-large  (OpenAI)       0.361
Cohere embed-v3                        0.350
lss embedding-only                     0.340
ColBERTv2                              0.338
text-embedding-3-small  (OpenAI)       0.336
lss hybrid                             0.334
BM25 (Anserini/Lucene)                 0.325
```

lss is competitive with ColBERTv2 and text-embedding-3-small. Hybrid is slightly below embedding-only here because FTS5's Porter stemmer doesn't handle medical terminology as well as Lucene's analyzer — medical compound terms get mis-stemmed, dragging BM25's contribution down.

---

## 4. Summary Table

All NDCG@10 scores side by side:

```
System                       Golden Set    SciFact    NFCorpus
──────────────────────────────────────────────────────────────
lss hybrid                      0.932       0.729       0.334
lss embedding-only              0.901       0.719       0.340
lss bm25-only                   0.888         —           —
text-embedding-3-large            —         0.735       0.361
Cohere embed-v3                   —         0.717       0.350
Voyage-2                          —         0.713         —
text-embedding-3-small            —         0.694       0.336
ColBERTv2                         —         0.693       0.338
BM25 (Anserini/Lucene)            —         0.665       0.325
```

---

## 5. Architecture Advantages

**Why lss scores well with a small model:**

| Factor | Impact |
|--------|--------|
| **Custom BM25 re-scoring** | FTS5's built-in `bm25()` produces flat scores on short passages. Our re-scorer with proper TF saturation and IDF weighting gives 4.4x NDCG improvement on keyword queries. |
| **Reciprocal Rank Fusion** | RRF is robust — it doesn't need score calibration between BM25 and embeddings, just rank positions. This makes the fusion stable across different corpora. |
| **Pseudo-relevance feedback** | Short or vague queries get automatic expansion from top BM25 results, improving recall without user effort. |
| **Jaccard + phrase boosts** | Post-fusion features catch exact matches that embedding similarity might under-rank. |
| **MMR diversity** | Prevents returning 5 near-identical chunks from the same file, improving effective recall. |

---

## 6. Known Limitations

- **FTS5 Porter stemmer** is less capable than Lucene's analyzer on domain-specific vocabulary (medical, legal). This reduces BM25's contribution to hybrid on specialized corpora like NFCorpus.
- **Single language** — FTS5 tokenizer is English-optimized. Non-Latin scripts may not tokenize well for BM25. Embedding search still works via OpenAI's multilingual models.
- **BEIR scores are pre-custom-BM25** — the SciFact and NFCorpus numbers above were measured before v0.4.0's custom BM25 re-scoring. Re-running should improve BM25-only and potentially hybrid scores.
- **256-dim vectors** — trading some embedding precision for speed and storage. Bumping to 512 or 1536 dims would likely improve embedding-only scores at the cost of larger DB and slower search.

---

## 7. Reproducing BEIR Benchmarks

Requires `ir-datasets` and `ranx`:

```bash
pip install ir-datasets ranx

# SciFact (300 queries, ~5 min, uses OpenAI API)
python -m pytest tests/test_beir.py -k "scifact" -v

# NFCorpus (323 queries, ~8 min)
python -m pytest tests/test_beir.py -k "nfcorpus" -v

# All BEIR tests
python -m pytest tests/test_beir.py -v
```

Published baseline scores sourced from the [BEIR leaderboard](https://github.com/beir-cellar/beir), [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), and respective model papers.
