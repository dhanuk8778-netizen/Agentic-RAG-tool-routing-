"""
Minimal in-memory vector store: brute-force cosine similarity search over
normalized embeddings. This is deliberately simple (an (N, D) tensor + a
matmul) -- it's a correct, exact-search stand-in for a production ANN index
(FAISS / Qdrant / pgvector); swap `VectorStore.search` for an ANN library
call once the corpus is large enough that brute-force matmul is the
bottleneck (see README "Scaling the vector store").
"""
from __future__ import annotations

import torch


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.vectors: torch.Tensor | None = None

    def build(self, ids: list[str], vectors: torch.Tensor) -> None:
        assert vectors.shape[1] == self.dim
        self.ids = ids
        self.vectors = vectors  # assumed already L2-normalized

    def search(self, query_vec: torch.Tensor, top_k: int = 10) -> list[tuple[str, float]]:
        """query_vec: (D,) or (1, D) normalized embedding."""
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        sims = (query_vec @ self.vectors.t()).squeeze(0)  # (N,)
        k = min(top_k, sims.shape[0])
        top_vals, top_idx = torch.topk(sims, k)
        return [(self.ids[i], float(v)) for v, i in zip(top_vals.tolist(), top_idx.tolist())]

    def __len__(self) -> int:
        return len(self.ids)
