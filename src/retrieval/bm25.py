"""
Classic BM25 (Robertson & Zaragoza, 2009) lexical retriever. Used three
ways in this project:
  1. A standalone sparse-retrieval baseline to compare P@5 against.
  2. A hybrid signal (score fusion with the dense bi-encoder).
  3. A cheap way to mine hard negatives for cross-encoder training (top-N
     BM25 hits that are *not* ground-truth relevant make much more useful
     negatives than random documents).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from src.data.corpus import Document
from src.data.tokenizer import _words


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.N = 0

    def fit(self, documents: list[Document]) -> "BM25":
        self.doc_ids = [d.doc_id for d in documents]
        for d in documents:
            toks = _words(d.text)
            self.doc_lens.append(len(toks))
            tf = Counter(toks)
            self.doc_freqs.append(tf)
            for term in tf:
                self.df[term] += 1
        self.N = len(documents)
        self.avg_doc_len = sum(self.doc_lens) / max(self.N, 1)
        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
        return self

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = _words(query)
        tf = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            f = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
            score += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = [(self.doc_ids[i], self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
