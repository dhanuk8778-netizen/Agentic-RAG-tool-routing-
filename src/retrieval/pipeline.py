"""
End-to-end RAG retrieval pipeline: dense (bi-encoder) or hybrid (dense +
BM25 score fusion) first-stage retrieval over the full corpus, narrowed to
a small candidate set, then cross-encoder reranking of those candidates
into the final top-k -- the standard "retrieve-then-rerank" architecture
used in production RAG systems.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.corpus import Document
from src.data.tokenizer import SimpleTokenizer
from src.retrieval.bi_encoder import BiEncoder
from src.retrieval.bm25 import BM25
from src.retrieval.cross_encoder import CrossEncoder
from src.retrieval.vector_store import VectorStore


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    dense_score: float
    rerank_score: float | None = None


class RAGPipeline:
    def __init__(
        self,
        documents: list[Document],
        tokenizer: SimpleTokenizer,
        bi_encoder: BiEncoder,
        cross_encoder: CrossEncoder | None = None,
        bm25: BM25 | None = None,
        hybrid_alpha: float = 0.5,
        device: torch.device | None = None,
    ):
        self.documents = {d.doc_id: d for d in documents}
        self.tokenizer = tokenizer
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.bm25 = bm25
        self.hybrid_alpha = hybrid_alpha  # weight on dense score in hybrid fusion
        self.device = device or torch.device("cpu")
        self.store = VectorStore(dim=bi_encoder.proj.out_features)
        self._index_built = False

    @torch.no_grad()
    def build_index(self, batch_size: int = 64) -> None:
        self.bi_encoder.eval()
        ids, vecs = [], []
        docs = list(self.documents.values())
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            input_ids = torch.tensor(
                [self.tokenizer.encode(d.text) for d in batch], dtype=torch.long, device=self.device
            )
            emb = self.bi_encoder.encode(input_ids)
            ids.extend([d.doc_id for d in batch])
            vecs.append(emb.cpu())
        self.store.build(ids, torch.cat(vecs, dim=0))
        self._index_built = True

    @torch.no_grad()
    def retrieve_dense(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        assert self._index_built, "call build_index() first"
        self.bi_encoder.eval()
        q_ids = torch.tensor([self.tokenizer.encode(query)], dtype=torch.long, device=self.device)
        q_emb = self.bi_encoder.encode(q_ids)[0].cpu()
        hits = self.store.search(q_emb, top_k=top_k)
        return [RetrievedDoc(doc_id=did, text=self.documents[did].text, dense_score=score) for did, score in hits]

    def retrieve_hybrid(self, query: str, top_k: int = 50, candidate_pool: int = 200) -> list[RetrievedDoc]:
        """Fuse normalized dense cosine similarity with normalized BM25 score."""
        assert self.bm25 is not None, "hybrid retrieval requires a fitted BM25 index"
        dense_hits = {r.doc_id: r.dense_score for r in self.retrieve_dense(query, top_k=candidate_pool)}
        bm25_hits = dict(self.bm25.search(query, top_k=candidate_pool))

        def _normalize(d: dict[str, float]) -> dict[str, float]:
            if not d:
                return d
            lo, hi = min(d.values()), max(d.values())
            span = max(hi - lo, 1e-9)
            return {k: (v - lo) / span for k, v in d.items()}

        dense_n = _normalize(dense_hits)
        bm25_n = _normalize(bm25_hits)
        all_ids = set(dense_n) | set(bm25_n)
        fused = {
            did: self.hybrid_alpha * dense_n.get(did, 0.0) + (1 - self.hybrid_alpha) * bm25_n.get(did, 0.0)
            for did in all_ids
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedDoc(doc_id=did, text=self.documents[did].text, dense_score=score) for did, score in ranked]

    @torch.no_grad()
    def rerank(self, query: str, candidates: list[RetrievedDoc], top_k: int = 5) -> list[RetrievedDoc]:
        assert self.cross_encoder is not None, "no cross-encoder loaded"
        self.cross_encoder.eval()
        pair_ids = torch.tensor(
            [self.tokenizer.encode_pair(query, c.text) for c in candidates], dtype=torch.long, device=self.device
        )
        scores = self.cross_encoder.score(pair_ids).cpu().tolist()
        for c, s in zip(candidates, scores):
            c.rerank_score = s
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_k]

    def query(self, query: str, top_k: int = 5, candidate_pool: int = 50, use_hybrid: bool = True, use_reranker: bool = True) -> list[RetrievedDoc]:
        candidates = (
            self.retrieve_hybrid(query, top_k=candidate_pool)
            if (use_hybrid and self.bm25 is not None)
            else self.retrieve_dense(query, top_k=candidate_pool)
        )
        if use_reranker and self.cross_encoder is not None:
            return self.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]
