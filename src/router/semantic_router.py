"""
Semantic router: a small Transformer encoder (mean-pooled) + a multi-label
sigmoid head over the tool registry. Multi-label rather than softmax
because a single query can legitimately require more than one tool (the
"multi-intent" case in the reported metric) -- softmax would force a
single winner and structurally couldn't represent "search docs AND
calculate" as one prediction.

Deployed behind a threshold (default 0.5): every tool whose predicted
probability clears the threshold is invoked, in descending-probability
order, by the agent loop (src/serving/agent_loop.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.retrieval.transformer_backbone import TransformerEncoder
from src.router.tools import TOOL_NAMES


class SemanticRouter(nn.Module):
    def __init__(self, tokenizer_vocab_size: int, num_tools: int = len(TOOL_NAMES),
                 dim: int = 64, num_layers: int = 2, num_heads: int = 4, ff_dim: int = 128,
                 max_len: int = 48, pad_id: int = 0):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=tokenizer_vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim, max_len=max_len, pad_id=pad_id,
        )
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_tools))
        self.tool_names = TOOL_NAMES[:num_tools]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.encode_mean_pooled(input_ids)
        return self.head(pooled)  # (N, num_tools) logits

    @torch.no_grad()
    def route(self, input_ids: torch.Tensor, threshold: float = 0.5) -> list[list[str]]:
        """Returns, per example, the list of tool names above threshold,
        ordered by descending predicted probability."""
        probs = torch.sigmoid(self.forward(input_ids))
        results = []
        for row in probs:
            selected = [(self.tool_names[i], p.item()) for i, p in enumerate(row) if p.item() >= threshold]
            selected.sort(key=lambda x: x[1], reverse=True)
            results.append([name for name, _ in selected] or [self.tool_names[row.argmax().item()]])
        return results
