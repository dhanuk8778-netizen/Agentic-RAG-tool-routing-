"""
Uniform affine (fake) quantization, used to simulate the "Q" in QLoRA:
freeze the base model's weights in reduced precision (int8 or int4) while
keeping LoRA adapters in full precision. This is a from-scratch, simplified
stand-in for bitsandbytes' NF4 double-quantization (which needs a compiled
CUDA extension not available in this offline CPU environment) -- the same
core idea (per-tensor/per-channel affine quantization, dequantize on the
fly for compute) applies, just with a simpler quantization scheme.

Trade-off measured directly in tests/benchmarks: quantized weights use
`bits/32` of the storage of an fp32 tensor, at the cost of a small
reconstruction error (and, in a real GPU kernel, a dequantization step
before each matmul -- not modeled here since CPU fp32 matmul is already
the reference path in this repo).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class QuantizedTensor:
    q_values: torch.Tensor    # integer codes, dtype=torch.int8
    scale: torch.Tensor       # per-channel (out_features,) or scalar
    zero_point: torch.Tensor
    num_bits: int
    orig_shape: torch.Size

    def dequantize(self) -> torch.Tensor:
        return (self.q_values.float() - self.zero_point) * self.scale

    def compression_ratio(self) -> float:
        """Storage ratio vs. fp32 (ignoring the small scale/zero_point overhead)."""
        return 32.0 / self.num_bits


def quantize_tensor(w: torch.Tensor, num_bits: int = 8, per_channel: bool = True) -> QuantizedTensor:
    """Affine-quantize `w` (assumed shape (out_features, in_features)) to
    `num_bits`, per-output-channel by default (standard practice -- each
    output channel gets its own scale, reducing error vs. a single
    tensor-wide scale)."""
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    dim = 1 if per_channel and w.dim() == 2 else None

    if dim is not None:
        w_min = w.min(dim=dim, keepdim=True).values
        w_max = w.max(dim=dim, keepdim=True).values
    else:
        w_min, w_max = w.min(), w.max()

    span = (w_max - w_min).clamp(min=1e-8)
    scale = span / (qmax - qmin)
    zero_point = qmin - w_min / scale

    q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax).to(torch.int8)
    return QuantizedTensor(q_values=q, scale=scale.squeeze(dim) if dim is not None else scale,
                            zero_point=zero_point.squeeze(dim) if dim is not None else zero_point,
                            num_bits=num_bits, orig_shape=w.shape)


def quantization_error(w: torch.Tensor, num_bits: int = 8, per_channel: bool = True) -> dict:
    qt = quantize_tensor(w, num_bits=num_bits, per_channel=per_channel)
    scale = qt.scale.unsqueeze(1) if per_channel and w.dim() == 2 else qt.scale
    zp = qt.zero_point.unsqueeze(1) if per_channel and w.dim() == 2 else qt.zero_point
    dequant = (qt.q_values.float() - zp) * scale
    mse = torch.mean((w - dequant) ** 2).item()
    rel_error = (torch.norm(w - dequant) / torch.norm(w).clamp(min=1e-8)).item()
    return {"num_bits": num_bits, "mse": mse, "relative_l2_error": rel_error, "compression_ratio": qt.compression_ratio()}


class QuantizedFrozenLinear(nn.Module):
    """Drop-in replacement for a frozen nn.Linear that stores its weight in
    quantized form and dequantizes on the fly in forward(). Used as the
    "Q" (quantized, frozen) half of a QLoRA-style adapted layer; combine
    with LoRALinear (src/peft/lora.py) wrapping this instead of a plain
    nn.Linear for the full QLoRA recipe -- see
    src/peft/finetune_router.py for the end-to-end wiring.
    """

    def __init__(self, base: nn.Linear, num_bits: int = 8):
        super().__init__()
        qt = quantize_tensor(base.weight.data, num_bits=num_bits, per_channel=True)
        self.register_buffer("q_values", qt.q_values)
        # qt.scale/zero_point are per-output-channel, shape (out_features,);
        # keep the trailing singleton dim so they broadcast against
        # q_values' (out_features, in_features) shape in forward().
        self.register_buffer("scale", qt.scale.reshape(-1, 1))
        self.register_buffer("zero_point", qt.zero_point.reshape(-1, 1))
        self.num_bits = num_bits
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        self.in_features = base.in_features
        self.out_features = base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ((self.q_values.float() - self.zero_point) * self.scale).to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)
