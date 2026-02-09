# LSS Search Quality Evaluation

## Methods

| Method | Description |
|--------|-------------|
| **BM25** | FTS5 + custom BM25 re-scoring (k1=1.2, b=0.75) |
| **Embedding (OpenAI)** | text-embedding-3-small (256d), cosine similarity |
| **Embedding (local)** | bge-small-en-v1.5 (384d) via fastembed, zero API calls |
| **Hybrid** | BM25 + embedding via RRF (k=60), Jaccard/phrase boosts, MMR. **Default mode.** |

## Metrics

| Metric | Meaning |
|--------|---------|
| **NDCG@10** | Ranking quality — rewards relevant results at top. Primary IR metric. |
| **MRR@10** | How quickly first relevant result appears. 1.0 = always rank 1. |
| **Recall@10** | Fraction of relevant docs found in top 10. |
| **MAP@10** | Mean average precision across recall levels. |

---

## 1. Golden Set

33-file synthetic project corpus. 40 hand-labeled queries with 3-level graded relevance.

### OpenAI (`text-embedding-3-small`, 256d)

```
Method         NDCG@5  NDCG@10   MRR@10  Recall@5  Recall@10    P@5   P@10   MAP@10       ms
─────────────────────────────────────────────────────────────────────────────────────────────────
bm25            0.870    0.885    0.988     0.845      0.893  0.480  0.255    0.809       95
embedding       0.844    0.886    0.988     0.788      0.917  0.445  0.265    0.784     7417
hybrid          0.896    0.914    1.000     0.887      0.936  0.505  0.270    0.834     6461
```

### Local (`bge-small-en-v1.5`, 384d)

```
Method         NDCG@5  NDCG@10   MRR@10  Recall@5  Recall@10    P@5   P@10   MAP@10       ms
─────────────────────────────────────────────────────────────────────────────────────────────────
bm25            0.870    0.885    0.988     0.845      0.893  0.480  0.255    0.809      125
embedding       0.848    0.894    1.000     0.777      0.923  0.440  0.267    0.783     3213
hybrid          0.888    0.911    1.000     0.864      0.931  0.495  0.268    0.834      812
```

### Provider comparison (hybrid)

| Provider | NDCG@10 | MRR@10 | MAP@10 | Latency |
|----------|---------|--------|--------|---------|
| OpenAI (256d) | 0.914 | 1.000 | 0.834 | 6461 ms |
| Local (384d) | 0.911 | 1.000 | 0.834 | 812 ms |
| Delta | -0.003 | 0.0 | 0.0 | **~8x faster** |

Local is within 0.3% of OpenAI on NDCG@10. Identical MRR and MAP. 8x faster (no network calls).

### Per-category (hybrid, OpenAI)

```
Category         Queries   NDCG@10   MRR@10
────────────────────────────────────────────
keyword              22     0.951     1.000
procedural            5     0.984     1.000
conceptual            9     0.921     1.000
multi_concept         2     0.901     1.000
short_vague           2     0.676     1.000
```

---

## 2. Hardware-Constrained Benchmarks

Local provider under Docker cgroup limits (`docker run --cpus=N --memory=Xg`).

| Profile | NDCG@10 | MRR@10 | MAP@10 | Hybrid ms | Embed ms |
|---------|---------|--------|--------|-----------|----------|
| Mac bare metal | 0.911 | 1.000 | 0.834 | 812 | 3,213 |
| Docker 2cpu/2GB | 0.910 | 1.000 | 0.831 | 1,886 | 39,558 |
| Docker 1cpu/1GB | 0.910 | 1.000 | 0.831 | 2,897 | 55,253 |
| Docker 1cpu/512MB | OOM | — | — | — | — |

Quality is identical (+-0.001) across all profiles. Same model = same vectors = same ranking. Only latency varies. Minimum 1GB RAM required (512MB OOMs during fastembed model load).

---

## 3. BEIR SciFact

5,183 biomedical docs, 300 queries. OpenAI provider.

| System | NDCG@10 |
|--------|---------|
| text-embedding-3-large | 0.735 |
| **lss hybrid** | **0.729** |
| Cohere embed-v3 | 0.717 |
| lss embedding-only | 0.719 |
| Voyage-2 | 0.713 |
| text-embedding-3-small | 0.694 |
| ColBERTv2 | 0.693 |
| BM25 (Anserini) | 0.665 |

## 4. BEIR NFCorpus

3,633 medical docs, 323 queries. OpenAI provider.

| System | NDCG@10 |
|--------|---------|
| text-embedding-3-large | 0.361 |
| Cohere embed-v3 | 0.350 |
| lss embedding-only | 0.340 |
| ColBERTv2 | 0.338 |
| text-embedding-3-small | 0.336 |
| **lss hybrid** | **0.334** |
| BM25 (Anserini) | 0.325 |

Hybrid slightly below embedding-only here because FTS5's Porter stemmer mis-stems medical terms.

---

## 5. Summary

```
System                       Golden Set    SciFact    NFCorpus
                           OAI    Local
──────────────────────────────────────────────────────────────
lss hybrid                0.914   0.911     0.729       0.334
lss embedding-only        0.886   0.894     0.719       0.340
lss bm25-only             0.885   0.885       —           —
text-embedding-3-large      —       —       0.735       0.361
Cohere embed-v3             —       —       0.717       0.350
ColBERTv2                   —       —       0.693       0.338
BM25 (Anserini)             —       —       0.665       0.325
```

## Limitations

- FTS5 Porter stemmer weaker than Lucene on domain-specific vocabulary (medical, legal)
- English-optimized tokenizer. Non-Latin BM25 may degrade (embedding search still works)
- 256d OpenAI vectors trade some precision for speed/storage
- BEIR with local provider not yet benchmarked

## Reproduce

```bash
lss eval                            # golden set, current provider
LSS_PROVIDER=local lss eval         # force local
LSS_PROVIDER=openai lss eval        # force OpenAI

# BEIR (requires ir-datasets + ranx)
pip install ir-datasets ranx
python -m pytest tests/test_beir.py -k "scifact" -v
```
