"""
Cross-encoder reranker: query and candidate document are concatenated into
a single [CLS] query [SEP] doc [SEP] sequence and jointly encoded, letting
every query token attend to every document token (and vice versa) before a
single relevance score is produced. This is strictly more expressive than
the bi-encoder's independent-embedding-then-cosine-similarity approach, at
the cost of needing one full forward pass per (query, doc) pair -- which is
exactly why it's used to *rerank* a small candidate set (e.g. the
bi-encoder's top 50) rather than to search the full corpus.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.retrieval.transformer_backbone import TransformerEncoder


class CrossEncoder(nn.Module):
    def __init__(self, tokenizer_vocab_size: int, dim: int = 96, num_layers: int = 3,
                 num_heads: int = 4, ff_dim: int = 256, max_len: int = 96, pad_id: int = 0):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=tokenizer_vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim, max_len=max_len, pad_id=pad_id,
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, pair_ids: torch.Tensor) -> torch.Tensor:
        """pair_ids: (N, L) encoded via tokenizer.encode_pair. Returns (N,) relevance logits."""
        cls = self.encoder.encode_cls(pair_ids)
        return self.head(cls).squeeze(-1)

    def score(self, pair_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(pair_ids))
