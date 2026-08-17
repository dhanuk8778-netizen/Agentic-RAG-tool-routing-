"""
Compact Transformer encoder shared by the bi-encoder (dense retrieval) and
cross-encoder (reranking) models. Deliberately small (configurable, default
~2-4 layers, 64-128 dim) so it trains from scratch on CPU in minutes on the
synthetic corpus -- the point is to demonstrate a real, from-scratch dense
retrieval + cross-encoder stack (embeddings, attention, contrastive
training, reranking) rather than to wrap a pretrained checkpoint that isn't
reachable from this offline environment.

Swap `TransformerEncoder` for a pretrained backbone (e.g. a HF
sentence-transformers model) in production without changing any downstream
retrieval/reranking code -- see README "Using real pretrained encoders".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """Token-embedding + sinusoidal-position + standard nn.TransformerEncoder
    stack, with mean-pooling and CLS-pooling heads available. Padding is
    masked out of both attention and pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        max_len: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token hidden states (N, L, D)."""
        pad_mask = input_ids == self.pad_id  # True where padded
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(x)

    def encode_mean_pooled(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(input_ids)  # (N, L, D)
        mask = (input_ids != self.pad_id).unsqueeze(-1).float()  # (N, L, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def encode_cls(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(input_ids)
        return hidden[:, 0]  # CLS token position
