import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .bm25 import BM25Index
from .preprocess import split_sentences, tokenize, tokenize_pairs

METHOD_LABELS = {
    "hybrid": "Hybrid (TextRank + TF-IDF)",
    "textrank": "TextRank",
    "tfidf": "TF-IDF",
    "bm25": "BM25 (query-focused)",
}


# ------------------------------------------------------------------ helpers
def _normalize(scores):
    scores = np.asarray(scores, dtype=float)
    rng = scores.max() - scores.min()
    if rng == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / rng


def _position_prior(n):
    if n == 1:
        return np.ones(1)
    return _normalize(1.0 / np.sqrt(1 + np.arange(n)))


def _length_prior(sentences):
    """Prefer medium-length sentences (about 22 words); very short or very long ones score lower."""
    words = np.array([len(s.split()) for s in sentences], dtype=float)
    return np.exp(-(((words - 22.0) / 18.0) ** 2))


# ------------------------------------------------------------------ scorers
def score_tfidf(token_lists, X):
    sums = np.asarray(X.sum(axis=1)).ravel()
    lengths = np.array([max(len(t), 1) for t in token_lists], dtype=float)
    return _normalize(sums / np.sqrt(lengths))


def score_textrank(sim):
    w = sim.copy()
    np.fill_diagonal(w, 0)
    graph = nx.from_numpy_array(w)
    try:
        pr = nx.pagerank(graph, max_iter=200)
    except Exception:
        pr = {i: 1.0 for i in range(len(w))}
    return _normalize([pr[i] for i in range(len(w))])


def score_bm25(token_lists, query_tokens):
    return _normalize(BM25Index(token_lists).scores(query_tokens))


# ------------------------------------------------------------------ MMR (vectorized)
def mmr_select(scores, sim, k, lam=0.7):
    """Maximal Marginal Relevance: balance relevance and novelty."""
    n = len(scores)
    k = min(k, n)
    first = int(np.argmax(scores))
    selected = [first]
    available = np.ones(n, dtype=bool)
    available[first] = False
    max_sim = sim[first].copy()

    while len(selected) < k:
        mmr = lam * scores - (1 - lam) * max_sim
        mmr[~available] = -np.inf
        nxt = int(np.argmax(mmr))
        selected.append(nxt)
        available[nxt] = False
        max_sim = np.maximum(max_sim, sim[nxt])
    return sorted(selected)


def top_keywords(vec, X, surface, n=10):
    weights = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    out, seen = [], set()
    for i in weights.argsort()[::-1]:
        word = surface[terms[i]].most_common(1)[0][0]
        if word not in seen:
            out.append(word)
            seen.add(word)
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------ main API
def summarize(text, method="hybrid", num_sentences=None, ratio=0.2,
              query=None, use_mmr=True, lam=0.7):
    t0 = time.perf_counter()
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method: {method}")

    sentences = split_sentences(text)
    empty = {"summary": "", "sentences": [], "selected": [], "scores": [],
             "keywords": [], "elapsed_ms": 0.0}
    if not sentences:
        return empty

    n = len(sentences)
    k = num_sentences if num_sentences else max(1, int(round(n * ratio)))
    k = min(int(k), n)

    # Tokenize once; keep original surface forms for readable keywords
    token_lists, surface = [], defaultdict(Counter)
    for s in sentences:
        pairs = tokenize_pairs(s)
        token_lists.append([stem for stem, _ in pairs])
        for stem, word in pairs:
            surface[stem][word] += 1

    if not any(token_lists):
        return {**empty, "summary": " ".join(sentences[:k]), "sentences": sentences,
                "selected": list(range(k))}

    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True)
    X = vec.fit_transform(token_lists)
    sim = cosine_similarity(X)  # computed once, reused by TextRank and MMR

    query_tokens = tokenize(query) if query and query.strip() else []

    if method == "tfidf":
        scores = 0.9 * score_tfidf(token_lists, X) + 0.1 * _position_prior(n)
    elif method == "textrank":
        scores = score_textrank(sim)
    elif method == "bm25":
        if not query_tokens:
            raise ValueError("BM25 method needs a query with meaningful words.")
        scores = score_bm25(token_lists, query_tokens)
    else:  # hybrid
        scores = (0.40 * score_textrank(sim)
                  + 0.35 * score_tfidf(token_lists, X)
                  + 0.15 * _position_prior(n)
                  + 0.10 * _length_prior(sentences))
        if query_tokens:  # optional query boost
            scores = 0.6 * scores + 0.4 * score_bm25(token_lists, query_tokens)

    scores = np.asarray(scores, dtype=float)
    if use_mmr:
        selected = mmr_select(scores, sim, k, lam)
    else:
        selected = sorted(np.argsort(scores)[::-1][:k].tolist())

    return {
        "summary": " ".join(sentences[i] for i in selected),
        "sentences": sentences,
        "selected": selected,
        "scores": scores.tolist(),
        "keywords": top_keywords(vec, X, surface),
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }