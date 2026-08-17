import torch
import torch.nn as nn

from src.peft.lora import LoRALinear, apply_lora, apply_qlora, count_trainable_params
from src.peft.quantization import QuantizedFrozenLinear, quantization_error, quantize_tensor


def test_lora_linear_matches_base_at_init():
    """B is zero-initialized, so a fresh LoRALinear must be numerically
    identical to the frozen base layer (the adapter starts as a no-op)."""
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_freezes_base_params():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_lora_reduces_trainable_param_count():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    full_trainable = sum(p.numel() for p in model.parameters())
    apply_lora(model, target_substrings=("0", "2"), r=4)
    trainable, total = count_trainable_params(model)
    assert trainable < full_trainable
    assert trainable < total


def test_merge_into_base_matches_lora_forward():
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B.add_(torch.randn_like(lora.lora_B) * 0.1)  # make the adapter non-trivial
    merged = lora.merge_into_base()
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), merged(x), atol=1e-5)


def test_quantize_tensor_compression_ratio():
    w = torch.randn(32, 16)
    qt8 = quantize_tensor(w, num_bits=8)
    qt4 = quantize_tensor(w, num_bits=4)
    assert qt8.compression_ratio() == 4.0
    assert qt4.compression_ratio() == 8.0


def test_quantization_error_decreases_with_more_bits():
    w = torch.randn(64, 32)
    err8 = quantization_error(w, num_bits=8)
    err4 = quantization_error(w, num_bits=4)
    assert err8["relative_l2_error"] < err4["relative_l2_error"]


def test_quantized_frozen_linear_reasonable_approximation():
    base = nn.Linear(16, 8)
    qlin = QuantizedFrozenLinear(base, num_bits=8)
    x = torch.randn(4, 16)
    out_base = base(x)
    out_q = qlin(x)
    # 8-bit quantization should be a close (not exact) approximation
    rel_err = (out_base - out_q).norm() / out_base.norm().clamp(min=1e-8)
    assert rel_err.item() < 0.15


def test_apply_qlora_freezes_and_wraps():
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    apply_qlora(model, target_substrings=("0", "2"), num_bits=4, r=2)
    trainable, total = count_trainable_params(model)
    assert 0 < trainable < total
    x = torch.randn(3, 16)
    out = model(x)  # must not raise
    assert out.shape == (3, 4)
