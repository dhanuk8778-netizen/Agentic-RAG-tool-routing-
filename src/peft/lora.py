"""
LoRA (Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language
Models") from scratch: wraps an existing nn.Linear, freezes its weight,
and adds a trainable low-rank decomposition `B @ A` (rank r << min(in,
out)) whose output is added to the frozen layer's output, scaled by
alpha/r. Only A and B are trained -- for a linear layer of shape
(out, in), this cuts trainable parameters from `in*out` to `r*(in+out)`,
the core PEFT (parameter-efficient fine-tuning) trick used to adapt large
frozen models cheaply.

`apply_lora` walks a model and swaps every nn.Linear matching a name
pattern for a LoRALinear wrapper in-place, freezing all *other* parameters
-- the standard "freeze base model, train only adapters" PEFT recipe.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features, out_features = base.in_features, base.out_features
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B initialized to zero so the adapter starts as a no-op (delta = 0),
        # matching the base model's behavior before any LoRA training.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * delta

    def merge_into_base(self) -> nn.Linear:
        """Fold the low-rank delta into the base weight for inference-time
        deployment with zero LoRA overhead (standard LoRA merge). Only
        supported when the base is a plain nn.Linear (merging into a
        quantized QLoRA base would require re-quantizing the merged
        weight, which defeats the point of shipping it quantized)."""
        if not isinstance(self.base, nn.Linear):
            raise TypeError("merge_into_base() requires a plain nn.Linear base; QLoRA adapters are kept separate at inference.")
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.base.weight + delta_w)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged

    def num_trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def _disable_transformer_fastpath(model: nn.Module) -> None:
    """nn.TransformerEncoder has an eval-mode fastpath that bypasses normal
    submodule __call__s and reads e.g. `layer.linear1.weight` directly for
    a fused CPU/CUDA kernel -- which breaks the moment linear1 has been
    replaced with a LoRALinear wrapper (AttributeError: no `.weight`).
    Registering a forward hook on each TransformerEncoderLayer is enough to
    make PyTorch's internal fastpath-eligibility check bail out and fall
    back to normal (LoRA-compatible) per-module execution.
    """
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.register_forward_hook(lambda m, i, o: o)


def apply_lora(model: nn.Module, target_substrings: tuple[str, ...] = ("head",), r: int = 8, alpha: int = 16, dropout: float = 0.0) -> nn.Module:
    """Freeze every parameter in `model`, then replace nn.Linear submodules
    whose dotted name contains any of `target_substrings` with a LoRALinear
    wrapper (trainable). Returns the same model, mutated in place.

    Note: nn.MultiheadAttention's internal fused forward reads
    `self_attn.out_proj.weight`/`.bias` directly rather than calling
    `out_proj(x)` as a module, so out_proj cannot be wrapped this way
    without reimplementing attention -- target the encoder's `linear1`/
    `linear2` (feed-forward) sublayers and the classifier `head` instead,
    which are both called as ordinary submodules and adapt cleanly.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def _recurse(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(sub in full_name for sub in target_substrings):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                _recurse(child, full_name)

    _recurse(model)
    _disable_transformer_fastpath(model)
    return model


def apply_qlora(model: nn.Module, target_substrings: tuple[str, ...] = ("head", "linear1", "linear2"),
                 num_bits: int = 4, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> nn.Module:
    """The full QLoRA recipe: freeze the whole model, then for each matched
    nn.Linear, quantize its frozen base weight to `num_bits` (see
    src/peft/quantization.py) AND wrap it with a trainable LoRA adapter --
    so the memory-heavy base weights sit in low precision while only the
    small A/B matrices are trained in full precision, exactly the trade-off
    QLoRA (Dettmers et al., 2023) is named for.
    """
    from src.peft.quantization import QuantizedFrozenLinear

    for p in model.parameters():
        p.requires_grad_(False)

    def _recurse(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(sub in full_name for sub in target_substrings):
                quantized = QuantizedFrozenLinear(child, num_bits=num_bits)
                setattr(module, name, LoRALinear(quantized, r=r, alpha=alpha, dropout=dropout))
            else:
                _recurse(child, full_name)

    _recurse(model)
    _disable_transformer_fastpath(model)
    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
