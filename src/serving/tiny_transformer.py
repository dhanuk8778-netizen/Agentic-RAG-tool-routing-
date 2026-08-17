"""
Minimal GPT-style causal decoder, built with explicit multi-head
self-attention (rather than nn.TransformerEncoder) specifically so it can
support incremental decoding with a key/value cache -- the mechanism
behind both the KV-cache-reuse benchmark (src/serving/kv_cache.py) and
speculative decoding (src/serving/speculative_decoding.py).

Trained from scratch (next-token prediction) on the project's own corpus
text in scripts/train_tiny_lm.py, so the whole serving stack is
demonstrated end-to-end offline, the same "no external pretrained
weights" constraint as the rest of this repo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PastKV = tuple[torch.Tensor, torch.Tensor]  # (key, value), each (B, num_heads, T, head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, past_kv: PastKV | None = None, use_cache: bool = False):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, T, head_dim)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # causal mask: PyTorch's is_causal=True uses TOP-LEFT alignment
        # when q_len != k_len (query i may attend only to keys 0..i),
        # which is wrong here -- with a KV cache, query i (the i-th NEW
        # token) must attend to the entire past cache PLUS new positions
        # 0..i, i.e. a BOTTOM-RIGHT-aligned causal mask. Build that
        # explicitly whenever there's more than one new query position;
        # for a single new query (the normal incremental-decode step) or
        # a from-scratch full prefill (past_kv=None, so T==S) the two
        # conventions coincide and the fast is_causal path is safe to use.
        if T > 1 and k.size(2) != T:
            S = k.size(2)
            offset = S - T
            attn_mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        else:
            is_causal = T > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        new_kv = (k, v) if use_cache else None
        return out, new_kv


class GPTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, past_kv: PastKV | None = None, use_cache: bool = False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, num_layers: int = 4, num_heads: int = 4,
                 ff_dim: int = 512, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([GPTBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, past_kv: list[PastKV] | None = None, use_cache: bool = False):
        B, T = input_ids.shape
        past_len = past_kv[0][0].size(2) if past_kv is not None else 0
        positions = torch.arange(past_len, past_len + T, device=input_ids.device)

        x = self.token_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)
        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, kv = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_past.append(kv)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_past

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
