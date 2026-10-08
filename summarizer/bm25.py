import math
from collections import Counter

import numpy as np


class BM25Index:
    """Okapi BM25 with the smoothed (Lucene-style) IDF, which is never negative."""

    def __init__(self, token_lists, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(token_lists)
        self.tf = [Counter(t) for t in token_lists]
        self.dl = np.array([len(t) for t in token_lists], dtype=float)
        mean_len = self.dl.mean() if self.N else 0.0
        self.avgdl = mean_len if mean_len > 0 else 1.0

        df = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.idf = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def scores(self, query_tokens):
        out = np.zeros(self.N)
        for i, counts in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
            total = 0.0
            for q in query_tokens:
                f = counts.get(q, 0)
                if f:
                    total += self.idf[q] * f * (self.k1 + 1) / (f + norm)
            out[i] = total
        return out