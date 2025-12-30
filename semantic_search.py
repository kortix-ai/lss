import os, re, time, sqlite3, json, numpy as np, hashlib, unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai import OpenAI

import kb_config  # ensures mode-specific env behavior is applied
from kb_config import OPENAI_MODEL, OPENAI_DIM  # keep model/dim in sync with VERSION_KEY
import kb_store

# CONFIG - USING WORKING VALUES FROM search_module.py 
SPAN_WORDS = int(os.getenv("KB_SPAN_WORDS", 220)) 
SPAN_STRIDE = int(os.getenv("KB_SPAN_STRIDE", 200)) 
SENT_WORDS = int(os.getenv("KB_SENT_WORDS", 60)) 
K_SQL = int(os.getenv("KB_K_SQL", 600)) 
K_FINAL = int(os.getenv("KB_K_FINAL", 20))
TOP_OAI = int(os.getenv("KB_TOP_OAI", 28))
OAI_TIMEOUT = float(os.getenv("OAI_TIMEOUT", 2.0))
JACCARD_THRESHOLD = float(os.getenv("JACCARD_THRESHOLD", 0.83))
RRF_K = int(os.getenv("RRF_K", 60))

# Find PRF expansion terms from top docs with original stopwords
# REMOVED: Hardcoded STOP set - now using DF-based filtering

# Tokenizer function from search_module.py
tok = lambda s: re.findall(r"[a-z0-9]+", (s or "").lower())

def _get_document_frequency_batch(terms, con):
    """Get document frequency for multiple terms in single query"""
    if not terms:
        return {}
    
    # Check hot-stop set first (tokens known to exceed threshold)
    result = {}
    remaining_terms = []
    for term in terms:
        if term in HOT_STOP_SET:
            result[term] = float('inf')  # Mark as high-DF
        else:
            cached_df = DF_CACHE.get(f"df_{term}")
            if cached_df is not None:
                result[term] = cached_df
            else:
                remaining_terms.append(term)
    
    if not remaining_terms:
        return result
    
    try:
        # Single IN query for all remaining terms
        placeholders = ','.join(['?' for _ in remaining_terms])
        rows = con.execute(f"SELECT term, doc FROM fts_vocab WHERE term IN ({placeholders})", remaining_terms).fetchall()
        
        # Fill results from query
        term_df_map = {term: df for term, df in rows}
        for term in remaining_terms:
            df = term_df_map.get(term, 0)
            result[term] = df
            DF_CACHE.put(f"df_{term}", df)
    except:
        # Fallback to 0 for all if fts5vocab fails
        for term in remaining_terms:
            result[term] = 0
            DF_CACHE.put(f"df_{term}", 0)
    
    return result

def _get_total_docs(con):
    """Get total document count with caching"""
    cached_total = DF_CACHE.get("total_docs")
    if cached_total is not None:
        return cached_total
    
    try:
        total = con.execute("SELECT COUNT(DISTINCT file_uid) FROM fts").fetchone()[0]
        DF_CACHE.put("total_docs", total)
        return total
    except:
        return 1000  # Safe fallback

def _detect_phrases(q):
    """Detect phrases from quotes and auto-extract contiguous bigrams/trigrams"""
    phrases = []
    
    # Extract quoted phrases (all quote types)  
    for pattern in [r'"([^"]+)"', r"'([^']+)'", r"'([^']+)'", r"'([^']+)'"]:
        phrases.extend(re.findall(pattern, q))
    
    # Auto-extract contiguous bigrams/trigrams from query
    words = tok(q)
    if len(words) >= 2:
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            bigram = " ".join(words[i:i+2])
            if len(bigram) > 5:  # Only meaningful bigrams
                phrases.append(bigram)
        # Extract 3-word phrases for longer queries
        if len(words) >= 3:
            for i in range(len(words) - 2):
                trigram = " ".join(words[i:i+3]) 
                if len(trigram) > 8:  # Only meaningful trigrams
                    phrases.append(trigram)
    
    return list(set(phrases))  # dedupe

def _keywords(q, max_terms=16, con=None, df_map=None):  # ENHANCED: Use pre-computed DF map
    terms = tok(q)
    nums = [t for t in terms if re.fullmatch(r"\d+(\.\d+)?", t)]
    
    # Detect protected phrases
    phrases = _detect_phrases(q)
    phrase_tokens = set()
    for phrase in phrases:
        phrase_tokens.update(tok(phrase))
    
    # Short-query fast path: ≤6 content tokens or contains quotes -> skip DF gating
    content_terms = [t for t in terms if t not in nums and len(t) > 2]
    has_quotes = '"' in q or "'" in q or "'" in q or "'" in q
    
    if len(content_terms) <= 6 or has_quotes:
        # Fast path: rely on phrase protection, minimal filtering
        words = [t for t in content_terms if len(t) > 2]
        words = sorted(set(words), key=lambda x: (-len(x), x))[:max_terms]
        return list(dict.fromkeys(nums + words))
    
    # Full DF-based filtering for longer queries
    words = []
    if con and df_map is not None:
        total_docs = _get_total_docs(con)
        df_threshold = max(total_docs * 0.15, 50)  # Lowered from 0.5 to 0.15
        
        for t in content_terms:
            if t in phrase_tokens:
                words.append(t)  # protect phrase tokens
                continue
                
            # Use pre-computed DF map
            df = df_map.get(t, 0)
            if df == float('inf'):  # Hot-stop token
                continue
            if df < df_threshold:  # Keep if not too common
                words.append(t)
            elif df >= df_threshold:
                HOT_STOP_SET.add(t)  # Add to hot-stop set for future queries
    else:
        # Fallback to length filtering if no connection
        words = content_terms
    
    # Adaptive max_terms based on query length
    adaptive_max = 12 if len(content_terms) <= 12 else max_terms
    
    # Sort by IDF estimate (length is rough proxy when no DF available)
    words = sorted(set(words), key=lambda x: (-len(x), x))[:adaptive_max]
    return list(dict.fromkeys(nums + words))

def _fts_or(keys):
    """Smart FTS query building with NEAR for auto-phrases"""
    if not keys:
        return None
    parts = []
    for k in keys:
        if re.fullmatch(r"\d+(\.\d+)?", k):
            parts.append(k)  # Numbers as-is
        else:
            parts.append(f'"{k}"')  # Words in quotes
    return " OR ".join(parts)

def _build_fts_with_phrases(keys, phrases):
    """Build FTS query with keys + NEAR phrases for better entity matching"""
    base_query = _fts_or(keys)
    if not phrases:
        return base_query
    
    # Add NEAR clauses for detected phrases (fixed syntax)
    phrase_parts = []
    for phrase in phrases:
        words = tok(phrase)
        if len(words) >= 2:
            # Use quoted phrase - NEAR syntax can be tricky, use simpler approach
            quoted_phrase = f'"{phrase}"'
            phrase_parts.append(quoted_phrase)
    
    if phrase_parts:
        phrase_query = " OR ".join(phrase_parts)
        return f"({base_query}) OR ({phrase_query})" if base_query else phrase_query
    
    return base_query

def _snippet(text, query, max_chars=280):  # FAST: Like search_module
    sents = re.split(r'(?<=[.!?])\s+', text)
    if not sents: return (text[:max_chars] + ("…" if len(text) > max_chars else ""))
    qt = set(tok(query))
    best = max(range(len(sents)), key=lambda i: sum(w in sents[i].lower() for w in qt))
    left, right = max(0, best-1), min(len(sents), best+3)
    out = " ".join(sents[left:right])
    return (out[:max_chars].rsplit(" ", 1)[0] + "…") if len(out) > max_chars else out

def _minmax(xs):
    if xs is None or len(xs) == 0: 
        return []
    if isinstance(xs, (list, tuple)):
        xs = np.array(xs)
    lo, hi = np.min(xs), np.max(xs)
    return [0.5] * len(xs) if hi - lo < 1e-9 else [(x - lo) / (hi - lo) for x in xs]

class LRU:
    def __init__(self, cap=2048, ttl=600):
        self.cap, self.ttl, self.d = cap, ttl, OrderedDict()
    
    def get(self, k):
        v = self.d.get(k)
        if not v: return None
        vec, ts = v
        if time.time() - ts > self.ttl:
            self.d.pop(k, None)
            return None
        self.d.move_to_end(k)
        return vec
    
    def put(self, k, v):
        self.d[k] = (v, time.time())
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

OAI_Q_CACHE = LRU(512, ttl=900)  
OAI_D_CACHE = LRU(8192, ttl=3600)  
DF_CACHE = LRU(8192, ttl=21600)  # Cache DF lookups for 6 hours
HOT_STOP_SET = set()  # In-memory set of tokens that exceed DF threshold  

def _text_hash(text):
    return hashlib.md5(text.encode('utf-8')).digest()  # Binary to match kb_store



# DB & OPENAI - Dynamic connection
_client = None

def _get_db_connection(db_path=None):
    if db_path is None:
        db_path = kb_store.get_db_path()
    con = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    # Performance pragmas for faster fts5vocab scans
    con.execute("PRAGMA cache_size=-256")  # 256MB cache
    con.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
    con.execute("PRAGMA temp_store=MEMORY")
    return con

def _cli():
    global _client
    if _client is None:
        api_key = os.getenv('OPENAI_API_KEY', '').strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set (required for search operations)")
        _client = OpenAI()
    return _client

def _oai_embed(texts, db_path=None):  # SAME as search_module
    if not texts:
        return None
        
    cli = _cli()
    
    # parallel microbatches for large requests
    if len(texts) > 32:
        batch_size = 32
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        def _embed_batch(batch):
            def _call():
                return cli.embeddings.create(model=OPENAI_MODEL, input=batch, dimensions=OPENAI_DIM)
            
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call)
                try:
                    r = fut.result(timeout=OAI_TIMEOUT)
                    V = np.asarray([d.embedding for d in r.data], dtype=np.float32)
                    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
                    return V
                except (TimeoutError, Exception):
                    return None
        
        # parallel execution of batches
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_embed_batch, batch) for batch in batches]
            results = []
            
            for future in futures:
                try:
                    result = future.result(timeout=OAI_TIMEOUT * 1.5)
                    if result is None:
                        return None  # any batch fails = whole call fails
                    results.append(result)
                except:
                    return None
        
        # concatenate all results
        all_vecs = np.concatenate(results, axis=0)
        return all_vecs
    
    # single batch - original logic
    def _call():
        return cli.embeddings.create(model=OPENAI_MODEL, input=texts, dimensions=OPENAI_DIM)
    
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            r = fut.result(timeout=OAI_TIMEOUT)
        except (TimeoutError, Exception):
            return None
    
    V = np.asarray([d.embedding for d in r.data], dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V

def _adaptive_budget(scores, default=80):
    if len(scores) < 10: return min(len(scores), default)
    r1, r5, r10 = scores[0], scores[min(4, len(scores)-1)], scores[min(9, len(scores)-1)]
    score_std = np.std(scores[:10]) if len(scores) >= 10 else 0.1
    if score_std < 0.02 or (r1 - r10) < 0.1: return 100
    if r1 - r5 > 0.4: return 70
    if r1 - r10 > 0.5: return 80
    return 90

def _should_prf(query, scores, content_tokens=None):
    """HARD-GATED PRF: only if S1 is flat OR query has ≤4 content tokens after DF-gating"""
    if content_tokens:
        # Use pre-computed content tokens
        content_words = [t for t in content_tokens if not re.fullmatch(r"\d+(\.\d+)?", t)]
        if len(content_words) <= 4: 
            return True  # short content query needs expansion
    else:
        # Fallback to length-based filtering
        tokens = [t for t in tok(query) if len(t) > 2]
        if len(tokens) <= 4: 
            return True
    
    if len(scores) < 10: 
        return False
    
    # Flat scores check (low variance in top-10)
    score_std = np.std(scores[:10])
    if score_std < 0.02:  # very flat
        return True
        
    return False  # Don't expand otherwise

def _extract_prf_terms(docs, query_tokens, total_docs=None, df_threshold=None):
    """Extract expansion terms using RM3-style relevance modeling with DF-based filtering"""
    term_scores = {}
    
    for doc_id, text, score in docs[:PRF_K]:
        doc_weight = 1.0 / (1.0 + abs(score)) if score < 0 else score
        doc_tokens = tok(text)
        tf_map = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        for term, tf in tf_map.items():
            # Basic filtering
            if len(term) < 3 or re.fullmatch(r"\d+", term) or term in query_tokens:
                continue
            
            # Hot-stop check
            if term in HOT_STOP_SET:
                continue
            
            # Use passed DF threshold if available
            if df_threshold and total_docs:
                # We'd need DF lookup here, but for PRF we can be more lenient
                # Skip only the most obvious hot-stops
                pass
            
            idf_est = 2.0 if tf <= 2 else 1.5 if tf <= 5 else 1.0
            term_score = doc_weight * tf * idf_est
            term_scores[term] = term_scores.get(term, 0) + term_score
    
    sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])
    expansion_terms = [t for t, s in sorted_terms[:PRF_M] if s > 0.1]
    return expansion_terms[:PRF_M]

def _build_expanded_query(orig_keys, expansion_terms):
    """Build ANCHORED FTS5 query with original + expansion terms"""
    if not orig_keys: return _fts_or(expansion_terms) if expansion_terms else None
    if not expansion_terms: return _fts_or(orig_keys)
    
    # Anchored expansion: top 3 anchors + ≤5 expansions
    anchors = _fts_or(orig_keys[:3])  # Top 3 anchor terms
    expansions = _fts_or(expansion_terms[:5])  # Top 5 expansion terms
    
    # Structure: (ANCHORS) OR (TOP_ANCHOR AND expansions) 
    if expansions and orig_keys:
        return f"({anchors}) OR ({orig_keys[0]} AND ({expansions}))"
    return anchors

def _has_factual_pattern(merged_text, query):
    """GENERIC: Check if merged text contains factual patterns for the query (no hardcoding)"""
    if not merged_text:
        return 0.0
    
    text_lower = merged_text.lower()
    query_lower = query.lower()
    boost = 0.0
    
    # GENERIC: Number proximity boost for queries containing numbers
    query_numbers = re.findall(r'\b\d+\b', query)
    if query_numbers:
        # Look for any numbers from query within text
        for num in query_numbers:
            if num in text_lower:
                boost += 0.05
    
    # GENERIC: Year presence boost 
    years = re.findall(r'\b\d{4}\b', query)
    for year in years:
        if year in text_lower:
            boost += 0.05
    
    # GENERIC: Phrase proximity boost
    phrases = _detect_phrases(query)
    for phrase in phrases:
        if phrase.lower() in text_lower:
            boost += 0.03
    
    return min(boost, 0.2)  # Cap boost

# SEMANTIC SEARCH - Main interface
def semantic_search(scope_path, sentences, db_path="semantic_search.db", indexed_only=False):
    """Search for sentences in a file or folder"""
    from pathlib import Path
    
    if not sentences:
        return []
    
    # Determine scope type
    path = Path(scope_path).resolve()
    kb_db_path = kb_store.get_db_path()
    
    # For files, get the file_uid. For directories, pass None (we'll filter by path prefix)
    file_uid = None
    if path.is_file():
        if indexed_only:
            # Check if indexed without indexing
            file_uid = kb_store.get_file_uid(scope_path)
            if not file_uid:
                # File not indexed, return empty results
                return [[] for _ in sentences]
        else:
            try:
                file_uid = kb_store.ensure_indexed(scope_path)
            except Exception as e:
                print(f"Failed to ingest {scope_path}: {e}")
                return []
    elif path.is_dir():
        # Directory - files should already be indexed by cmd_search (or we're in indexed-only mode)
        pass
    else:
        print(f"Invalid scope path: {scope_path}")
        return []
    
    con = _get_db_connection(kb_db_path)
    cur = con.cursor()
    
    results = []
    seen = set()
    unique_sentences = []
    for sent in sentences:
        if sent not in seen:
            seen.add(sent)
            unique_sentences.append(sent)
    
    for sentence in unique_sentences:
        sentence_results = _search_single_sentence(sentence, file_uid, str(path), cur, con, kb_db_path)
        results.append(sentence_results)
    
    con.close()
    return results

# PRF CONFIG (copied from search_module)
PRF_K = int(os.getenv("PRF_K", "10"))  # feedback docs
PRF_M = int(os.getenv("PRF_M", "20"))  # expansion terms (slightly richer)
PRF_ALPHA = float(os.getenv("PRF_ALPHA", "0.7"))  # original query weight
K_SQL2 = int(os.getenv("K_SQL2", str(K_SQL + 300)))  # second pass limit (more generous)

def _jaccard_dedup(texts, scores, threshold=JACCARD_THRESHOLD):
    """ENHANCED: Remove near-duplicate texts using Jaccard similarity (ace code)"""
    if len(texts) <= 1:
        return list(range(len(texts)))
    
    word_sets = [set(tok(t)) for t in texts]
    order = np.argsort(-np.asarray(scores))
    keep = []
    
    for i in order:
        should_keep = True
        for j in keep:
            jaccard = len(word_sets[i] & word_sets[j]) / max(1, len(word_sets[i] | word_sets[j]))
            if jaccard >= threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)
    
    return keep

def _compute_boost_features(texts, query):
    """ENHANCED: Compute boost features for post-fusion enhancement (no stopword filtering)"""
    qt = [t for t in tok(query) if len(t) > 2]  # Use length filter instead of STOP
    qs = set(qt)
    qnums = set(re.findall(r'\d+', ' '.join(qt)))
    phrases = {' '.join(qt[i:i+n]) for n in (2, 3) for i in range(len(qt)-n+1)} or {''}
    
    jaccard_scores, phrase_scores, digit_scores = [], [], []
    
    for txt in texts:
        tt = tok(txt)
        s = ' '.join(tt)
        w = set(tt)
        
        # Jaccard similarity
        jaccard_scores.append(len(qs & w) / max(1, len(qs | w)))
        
        # Phrase matching
        phrase_scores.append(sum(1 for p in phrases if p and p in s) / max(1, len(phrases)))
        
        # Digit matching
        digit_scores.append(1 if qnums & set(re.findall(r'\d+', s)) else 0)
    
    # Normalize features
    def normalize(arr):
        arr = np.asarray(arr, float)
        return (arr - np.min(arr)) / (np.ptp(arr) + 1e-9)
    
    return normalize(jaccard_scores), normalize(phrase_scores), normalize(digit_scores)

def _compute_rank_correlation(bm25_ranks, embed_ranks):
    """Compute rank correlation between BM25 and embedding rankings"""
    if len(bm25_ranks) < 5:  # Need minimum data points
        return 0.0
    
    # Spearman rank correlation (simplified)
    n = len(bm25_ranks)
    sum_d_sq = sum((bm25_ranks[i] - embed_ranks[i]) ** 2 for i in range(n))
    correlation = 1 - (6 * sum_d_sq) / (n * (n**2 - 1))
    return correlation

def _search_single_sentence(q, file_uid, scope_path, cur, cache_con, db_path=None):
    """Search logic with PRF, adaptive budget, and speed optimizations"""
    from pathlib import Path
    
    t_all = time.time()
    version = kb_store.VERSION_KEY
    
    # COMPUTE ONCE: Tokenize and get DF map for all candidate terms
    all_terms = tok(q)
    content_terms = [t for t in all_terms if len(t) > 2 and not re.fullmatch(r"\d+(\.\d+)?", t)]
    
    # Batch DF lookup for all candidate terms
    df_map = _get_document_frequency_batch(content_terms, cur) if content_terms else {}
    
    # S1: BM25 with scope filter (file_uid for files, path prefix for directories)
    t0 = time.time()
    keys = _keywords(q, max_terms=16, con=cur, df_map=df_map)
    main_query = _fts_or(keys)
    
    rows = []
    if main_query:
        if file_uid is not None:
            # File scope - filter by file_uid with indexed_at
            rows = cur.execute(
                """SELECT fts.id, fts.text, bm25(fts) AS r, fts.text_hash, files.indexed_at 
                   FROM fts LEFT JOIN files ON fts.file_uid = files.file_uid 
                   WHERE fts MATCH ? AND fts.file_uid = ? ORDER BY r LIMIT ?""",
                (main_query, file_uid, K_SQL)
            ).fetchall()
        else:
            # Directory scope - filter by file_path prefix with indexed_at
            path_prefix = str(Path(scope_path).resolve()) + "/"
            rows = cur.execute(
                """SELECT fts.id, fts.text, bm25(fts) AS r, fts.text_hash, fts.file_path, files.indexed_at 
                   FROM fts LEFT JOIN files ON fts.file_uid = files.file_uid 
                   WHERE fts MATCH ? AND (fts.file_path = ? OR fts.file_path LIKE ?) ORDER BY r LIMIT ?""",
                (main_query, scope_path, path_prefix + "%", K_SQL)
            ).fetchall()
    
    bm25_ms = round((time.time() - t0) * 1000, 2)
    if not rows:
        if kb_config.DEBUG:
            print(f"LAT | S1_sql={bm25_ms}ms | 0 hits")
        return []

    # Normalize rows - handle both file scope (5 cols) and dir scope (6 cols) with indexed_at
    normalized_rows = []
    for row in rows:
        if len(row) == 5:
            # File scope: id, text, r, text_hash, indexed_at
            span_id, text, raw_score, text_hash, indexed_at = row
            normalized_rows.append((span_id, text, raw_score, text_hash, scope_path, indexed_at))
        else:  # len(row) == 6 (directory scope)
            # Dir scope: id, text, r, text_hash, file_path, indexed_at
            span_id, text, raw_score, text_hash, file_path, indexed_at = row
            normalized_rows.append((span_id, text, raw_score, text_hash, file_path, indexed_at))
    
    # PRF - RESTORE THIS LOGIC
    initial_docs = [(span_id, text, 1.0/(1.0+raw_score)) for span_id, text, raw_score, _, _, _ in normalized_rows]
    initial_scores = [score for _, _, score in initial_docs]
    
    # Hard-gated PRF check (reuse computed content tokens)
    query_tokens = set(keys)
    use_prf = _should_prf(q, initial_scores, content_tokens=keys)
    
    if use_prf:
        total_docs = _get_total_docs(cur)
        df_threshold = max(total_docs * 0.15, 50)  # Match _keywords threshold
        expansion_terms = _extract_prf_terms(initial_docs, query_tokens, total_docs, df_threshold)
        if expansion_terms:
            expanded_query = _build_expanded_query(keys, expansion_terms)
            if expanded_query:
                t_prf = time.time()
                if file_uid is not None:
                    # File scope with indexed_at
                    prf_rows = cur.execute(
                        """SELECT fts.id, fts.text, bm25(fts) AS r, fts.text_hash, files.indexed_at 
                           FROM fts LEFT JOIN files ON fts.file_uid = files.file_uid 
                           WHERE fts MATCH ? AND fts.file_uid = ? ORDER BY r LIMIT ?""",
                        (expanded_query, file_uid, K_SQL2)
                    ).fetchall()
                else:
                    # Directory scope with indexed_at
                    path_prefix = str(Path(scope_path).resolve()) + "/"
                    prf_rows = cur.execute(
                        """SELECT fts.id, fts.text, bm25(fts) AS r, fts.text_hash, fts.file_path, files.indexed_at 
                           FROM fts LEFT JOIN files ON fts.file_uid = files.file_uid 
                           WHERE fts MATCH ? AND (fts.file_path = ? OR fts.file_path LIKE ?) ORDER BY r LIMIT ?""",
                        (expanded_query, scope_path, path_prefix + "%", K_SQL2)
                    ).fetchall()
                
                if prf_rows:
                    # Normalize prf_rows with indexed_at
                    normalized_prf_rows = []
                    for row in prf_rows:
                        if len(row) == 5:
                            # File scope: id, text, r, text_hash, indexed_at
                            span_id, text, raw_score, text_hash, indexed_at = row
                            normalized_prf_rows.append((span_id, text, raw_score, text_hash, scope_path, indexed_at))
                        else:  # len(row) == 6
                            # Dir scope: id, text, r, text_hash, file_path, indexed_at
                            span_id, text, raw_score, text_hash, file_path, indexed_at = row
                            normalized_prf_rows.append((span_id, text, raw_score, text_hash, file_path, indexed_at))
                    
                    # Drift protection: check overlap ≥0.4
                    orig_top10 = set(doc_id for doc_id, _, _ in initial_docs[:10])
                    prf_top10 = set(span_id for span_id, _, _, _, _, _ in normalized_prf_rows[:10])
                    overlap = len(orig_top10 & prf_top10) / 10.0 if orig_top10 else 0.0
                    
                    if overlap >= 0.4:  # drift guard passed
                        normalized_rows = normalized_prf_rows
                        if kb_config.DEBUG:
                            prf_ms = round((time.time() - t_prf) * 1000, 2)
                            print(f"PRF | exp_terms={len(expansion_terms)} | overlap={overlap:.2f} | {prf_ms}ms")
                    else:
                        if kb_config.DEBUG:
                            print(f"PRF | DRIFT | overlap={overlap:.2f} < 0.4 | fallback")

    # Slice to TOP_OAI and dedupe 
    ids = [i for i, _, _, _, _, _ in normalized_rows[:TOP_OAI]]
    texts = [t for _, t, _, _, _, _ in normalized_rows[:TOP_OAI]]
    text_hashes = [h for _, _, _, h, _, _ in normalized_rows[:TOP_OAI]]
    file_paths = [fp for _, _, _, _, fp, _ in normalized_rows[:TOP_OAI]]
    indexed_ats = [ia for _, _, _, _, _, ia in normalized_rows[:TOP_OAI]]
    braw = [1.0/(1.0+float(r)) for _, _, r, _, _, _ in normalized_rows[:TOP_OAI]]
    bnorm = _minmax(braw)

    if len(texts) > 1:
        keep_indices = _jaccard_dedup(texts, bnorm, JACCARD_THRESHOLD)
        ids = [ids[k] for k in keep_indices]
        texts = [texts[k] for k in keep_indices]
        text_hashes = [text_hashes[k] for k in keep_indices]
        file_paths = [file_paths[k] for k in keep_indices]
        indexed_ats = [indexed_ats[k] for k in keep_indices]
        bnorm = [bnorm[k] for k in keep_indices]

    # Skip embedding for pure numeric queries (reuse computed keys)
    nums = [t for t in keys if re.fullmatch(r"\d{3,4}", t)]
    content_words = [t for t in keys if not re.fullmatch(r"\d{3,4}", t)]
    if len(nums) > 0 and len(content_words) == 0:
        out = []
        for j in range(min(K_FINAL, len(ids))):
            # Extract file_uid from chunk_id or use file_uid if available
            chunk_file_uid = ids[j].split("::")[0] if "::" in ids[j] else (file_uid or ids[j])
            out.append({
                "file_uid": chunk_file_uid,
                "file_path": file_paths[j],
                "chunk_id": ids[j],
                "score": float(bnorm[j]),
                "snippet": _snippet(texts[j], q),
                "rank_stage": "S1",
                "indexed_at": indexed_ats[j]
            })
        return out

    # Use adaptive budget for rerank pool, but cap at available candidates
    adaptive_budget = _adaptive_budget(initial_scores)
    rerank_pool = min(adaptive_budget, len(ids))  # Cap at available after dedup
    
    # Slice to rerank pool
    ids = ids[:rerank_pool] 
    texts = texts[:rerank_pool]
    text_hashes = text_hashes[:rerank_pool]
    file_paths = file_paths[:rerank_pool]

    # S2: SINGLE OpenAI call
    t2 = time.time()
    
    embed_texts = texts  # Use full texts, not tiered
    
    # Batch check persistent cache for all docs at once
    sims_o = [0.0] * len(ids)
    need_idx, need_texts, need_hashes = [], [], []
    
    if text_hashes:
        # Single SELECT for all hashes
        placeholders = ','.join(['?' for _ in text_hashes])
        cache_rows = cache_con.execute(
            f"SELECT text_hash, vector FROM embeddings WHERE text_hash IN ({placeholders}) AND model=? AND dim=? AND version=?",
            text_hashes + [OPENAI_MODEL, OPENAI_DIM, version]
        ).fetchall()
        cache_map = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cache_rows}
        
        for i, (text_hash, embed_text) in enumerate(zip(text_hashes, embed_texts)):
            cached_v = cache_map.get(text_hash)
            if cached_v is None:
                cached_v = OAI_D_CACHE.get(("d", text_hash, OPENAI_MODEL, OPENAI_DIM))
            
            if cached_v is not None:
                cache_map[text_hash] = cached_v  # ensure it's in cache_map for query dot product
            else:
                need_idx.append(i)
                need_texts.append(embed_text)
                need_hashes.append(text_hash)
    
    # SINGLE combined embedding call: [query] + need_texts  
    embed_input = [q] + need_texts
    V = _oai_embed(embed_input) if embed_input else None
    
    if V is not None:
        qv = V[0]  # query vector
        
        # Batch store new doc vectors
        if need_idx:
            new_cache_data = []
            for j, (i, text_hash) in enumerate(zip(need_idx, need_hashes)):
                dv = V[1 + j]  # doc vectors start at index 1
                cache_map[text_hash] = dv
                OAI_D_CACHE.put(("d", text_hash, OPENAI_MODEL, OPENAI_DIM), dv)
                new_cache_data.append((text_hash, OPENAI_MODEL, OPENAI_DIM, version, dv.tobytes(), time.time()))
            
            # Single executemany + commit for all new vectors
            if new_cache_data:
                cache_con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)", new_cache_data)
                cache_con.commit()
        
        # Compute similarities using cached vectors
        for i, text_hash in enumerate(text_hashes):
            dv = cache_map.get(text_hash)
            if dv is not None:
                sims_o[i] = float(dv @ qv)
    else:
        # Embedding failed, use BM25 only
        out = []
        for j in range(min(K_FINAL, len(ids))):
            chunk_file_uid = ids[j].split("::")[0] if "::" in ids[j] else (file_uid or ids[j])
            out.append({
                "file_uid": chunk_file_uid,
                "file_path": file_paths[j],
                "chunk_id": ids[j],
                "score": float(bnorm[j]),
                "snippet": _snippet(texts[j], q),
                "rank_stage": "S1_embed_fail",
                "indexed_at": indexed_ats[j]
            })
        return out
    
    # RRF Fusion with ADAPTIVE embedding weight
    onorm = _minmax(sims_o) if any(sims_o) else [0.0] * len(ids)
    sims_std = np.std(sims_o) if sims_o else 0.0
    
    # Adaptive embedding weight based on BM25-embedding rank correlation
    bm25_ranks = list(range(len(ids)))
    embed_pairs = [(i, sims_o[i]) for i in range(len(ids))]
    embed_pairs.sort(key=lambda x: -x[1])
    embed_ranks = [0] * len(ids)
    for rank, (idx, _) in enumerate(embed_pairs):
        embed_ranks[idx] = rank
    
    # Set embedding weight
    embedding_weight = 1.0/(1.0 + np.exp(-(sims_std - 0.008)/0.004))
    embedding_weight = 0.6 + 0.4*embedding_weight
    
    # RRF scoring
    embed_pairs = [(i, onorm[i] * embedding_weight) for i in range(len(ids))]
    embed_pairs.sort(key=lambda x: -x[1])
    embed_ranks = [0] * len(ids)
    for rank, (idx, _) in enumerate(embed_pairs):
        embed_ranks[idx] = rank
    
    rrf_scores = []
    for j in range(len(ids)):
        bm25_rrf = 1.0 / (RRF_K + bm25_ranks[j])
        embed_rrf = 1.0 / (RRF_K + embed_ranks[j])
        rrf_score = bm25_rrf + embed_rrf
        rrf_scores.append(rrf_score)
    
    # POST-FUSION CO-MENTION BOOST (restore this)
    jaccard_boost, phrase_boost, digit_boost = _compute_boost_features(texts, q)
    
    final_scores = []
    for i in range(len(rrf_scores)):
        base = rrf_scores[i]
        jacc, phrb, digb = jaccard_boost[i], phrase_boost[i], digit_boost[i]
        co_mention = 0.4*jacc + 0.3*phrb + 0.1*digb
        final_scores.append(base * (1 + co_mention))
    
    # Sort by enhanced scores
    scored_docs = list(zip(ids, texts, final_scores))
    scored_docs.sort(key=lambda x: x[2], reverse=True)
    
    # VECTOR-MMR with λ≈0.7, GATED by ≥85% vector coverage  
    nonzero_sims = sum(1 for s in sims_o if s > 0)
    vector_coverage = nonzero_sims / len(ids) if ids else 0.0
    
    if len(scored_docs) > K_FINAL and vector_coverage >= 0.90:  # REVERT: Back to 0.90
        mmr_selected = []
        remaining = list(scored_docs)
        
        # Always take top result
        mmr_selected.append(remaining.pop(0))
        
        # Vector-MMR selection with λ=0.7
        lambda_mmr = 0.7  # balance relevance vs diversity
        while len(mmr_selected) < K_FINAL and remaining:
            best_idx = 0
            best_score = -float('inf')
            
            for i, (doc_id, text, rel_score) in enumerate(remaining):
                doc_hash = text_hashes[ids.index(doc_id)] if doc_id in ids else _text_hash(text)
                doc_vec = cache_map.get(doc_hash)
                
                if doc_vec is not None:
                    relevance = float(doc_vec @ qv)
                    
                    # Max similarity with already selected
                    max_sim = 0.0
                    for sel_id, sel_text, _ in mmr_selected:
                        sel_hash = text_hashes[ids.index(sel_id)] if sel_id in ids else _text_hash(sel_text)
                        sel_vec = cache_map.get(sel_hash)
                        if sel_vec is not None:
                            sim = float(doc_vec @ sel_vec)
                            max_sim = max(max_sim, sim)
                    
                    # MMR score: λ * relevance - (1-λ) * max_similarity
                    mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * max_sim
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
            
            mmr_selected.append(remaining.pop(best_idx))
        
        final_docs = mmr_selected
    else:
        final_docs = scored_docs[:K_FINAL]
    
    # Build output
    oai_ms = round((time.time() - t2) * 1000, 2)
    out = []
    for i, (doc_id, text, final_score) in enumerate(final_docs):
        # Find the file_path and indexed_at for this doc_id
        doc_idx = ids.index(doc_id)
        doc_file_path = file_paths[doc_idx]
        doc_indexed_at = indexed_ats[doc_idx]
        chunk_file_uid = doc_id.split("::")[0] if "::" in doc_id else (file_uid or doc_id)
        out.append({
            "file_uid": chunk_file_uid,
            "file_path": doc_file_path,
            "chunk_id": doc_id,
            "score": float(final_score), 
            "snippet": _snippet(text, q),
            "rank_stage": "S3_MMR" if vector_coverage >= 0.90 else "S3",
            "indexed_at": doc_indexed_at
        })
    
    cache_hits = len(ids) - len(need_idx) if need_idx else len(ids)
    cache_pct = round(100 * cache_hits / len(ids), 1) if ids else 0
    
    if kb_config.DEBUG:
        df_hits = len([t for t in content_terms if t not in HOT_STOP_SET])
        print(f"LAT | docs={len(out)} | rerank={rerank_pool} | df_batch={len(content_terms)} | df_hits={df_hits} | embeds={nonzero_sims}/{len(ids)} | embed_batch={len(need_texts)} | cache={cache_pct}% | vec_cov={vector_coverage:.1%} | {round((time.time()-t_all)*1000,1)}ms")
    return out

__all__ = ["semantic_search"]
