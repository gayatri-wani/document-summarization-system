import numpy as np

from .bm25 import BM25Index
from .preprocess import split_sentences, tokenize


class DocumentIndex:
    """Tiny search engine: BM25 over documents, plus best-sentence snippets."""

    def __init__(self, docs: dict):
        self.names = list(docs.keys())
        self.texts = [docs[n] for n in self.names]
        self.index = BM25Index([tokenize(t) for t in self.texts])

    def _snippet(self, text, query_tokens):
        sentences = split_sentences(text)
        if not sentences:
            return text[:200]
        scores = BM25Index([tokenize(s) for s in sentences]).scores(query_tokens)
        return sentences[int(np.argmax(scores))]

    def search(self, query, top_k=5):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.index.scores(query_tokens)
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in order:
            if scores[i] <= 0:
                continue
            results.append({
                "name": self.names[i],
                "text": self.texts[i],
                "score": float(scores[i]),
                "snippet": self._snippet(self.texts[i], query_tokens),
            })
        return results