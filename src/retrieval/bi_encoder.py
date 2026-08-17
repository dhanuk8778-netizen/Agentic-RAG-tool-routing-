"""
Dense bi-encoder for retrieval: a shared TransformerEncoder embeds queries
and documents independently into a common vector space, trained with an
in-batch-negatives contrastive (InfoNCE) objective -- the standard recipe
behind dense retrievers like DPR / Sentence-BERT.

Given a batch of B (query, positive_doc) pairs, every other document in the
batch is treated as a negative for that query, giving B-1 free negatives
per example with a single forward pass -- the standard trick that makes
contrastive retriever training cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.retrieval.transformer_backbone import TransformerEncoder


class BiEncoder(nn.Module):
    def __init__(self, tokenizer_vocab_size: int, dim: int = 96, num_layers: int = 3,
                 num_heads: int = 4, ff_dim: int = 256, max_len: int = 64, pad_id: int = 0,
                 proj_dim: int = 64):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=tokenizer_vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim, max_len=max_len, pad_id=pad_id,
        )
        self.proj = nn.Linear(dim, proj_dim)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.encode_mean_pooled(input_ids)
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)

    def forward(self, query_ids: torch.Tensor, doc_ids: torch.Tensor):
        return self.encode(query_ids), self.encode(doc_ids)


def in_batch_contrastive_loss(query_emb: torch.Tensor, doc_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """InfoNCE with in-batch negatives. query_emb[i] should match doc_emb[i];
    all other doc_emb[j] in the batch serve as negatives for query i."""
    logits = query_emb @ doc_emb.t() / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_q2d = F.cross_entropy(logits, labels)
    loss_d2q = F.cross_entropy(logits.t(), labels)
    return (loss_q2d + loss_d2q) / 2
