"""
Demonstrates parameter-efficient fine-tuning (QLoRA-style: quantized frozen
base + trainable low-rank adapters) by adapting the semantic router -- pretrained
on full-sentence queries -- to a shifted, terse "Slack-shorthand" query
distribution it was never trained on.

Compares three configurations on the shifted domain's held-out eval split:
  1. zero-shot: pretrained router, no adaptation at all.
  2. full fine-tune: every parameter trainable (the expensive baseline).
  3. QLoRA: base weights quantized to 4-bit and frozen; only small LoRA
     adapters (attached to the encoder's feed-forward + attention output
     projections and the classifier head) are trained.

Reports accuracy for all three plus the trainable-parameter-count and
memory-footprint gap between (2) and (3) -- the actual point of PEFT.

Usage:
    python -m src.peft.finetune_router --router-config configs/router_config.yaml
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.data.router_queries import generate_shifted_domain_queries
from src.data.tokenizer import SimpleTokenizer
from src.peft.lora import apply_lora, apply_qlora, count_trainable_params
from src.peft.quantization import quantization_error
from src.router.semantic_router import SemanticRouter
from src.train_router import evaluate_router, labels_to_multihot
from src.router.llm_judge import RuleBasedJudge
from src.utils import save_json, set_seed, timer


def finetune(model, tokenizer, examples, device, epochs, batch_size, lr):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        total_loss, n_batches = 0.0, 0
        for i in range(0, max(len(examples) - batch_size + 1, 1), batch_size):
            batch = examples[i : i + batch_size]
            if not batch:
                continue
            input_ids = torch.tensor([tokenizer.encode(ex.text) for ex in batch], dtype=torch.long, device=device)
            labels = torch.tensor([labels_to_multihot(ex.tool_labels) for ex in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"    epoch {epoch:02d} | loss {avg:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-config", type=str, default="configs/router_config.yaml")
    parser.add_argument("--router-checkpoint", type=str, default="checkpoints/semantic_router.pt")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/router_tokenizer.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--num-bits", type=int, default=4)
    args = parser.parse_args()

    with open(args.router_config) as f:
        router_cfg = yaml.safe_load(f)
    set_seed(router_cfg["seed"])
    device = torch.device("cpu")

    tokenizer = SimpleTokenizer.load(args.tokenizer_path)

    def load_base_model():
        m = SemanticRouter(tokenizer_vocab_size=tokenizer.vocab_size, **router_cfg["model"])
        m.load_state_dict(torch.load(args.router_checkpoint, map_location=device))
        return m

    with timer("shifted-domain query generation"):
        shifted = generate_shifted_domain_queries(num_queries=150, seed=99)
    split = int(len(shifted) * 0.5)
    shift_train, shift_eval = shifted[:split], shifted[split:]
    print(f"shifted-domain: train={len(shift_train)} eval={len(shift_eval)}")

    judge = RuleBasedJudge()  # deterministic, so the three configs are compared fairly

    results = {}

    print("\n=== [1/3] Zero-shot (no adaptation) ===")
    zero_shot_model = load_base_model()
    m = evaluate_router(zero_shot_model, tokenizer, shift_eval, device, threshold=router_cfg["eval"]["threshold"], judge=judge)
    print(f"exact_match={m['exact_match_accuracy']:.3f}  jaccard={m['jaccard_similarity']:.3f}")
    results["zero_shot"] = {"exact_match_accuracy": m["exact_match_accuracy"], "jaccard_similarity": m["jaccard_similarity"]}

    print("\n=== [2/3] Full fine-tuning (all parameters trainable) ===")
    full_ft_model = load_base_model()
    trainable, total = count_trainable_params(full_ft_model)
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    with timer("full fine-tune"):
        finetune(full_ft_model, tokenizer, list(shift_train), device, epochs=args.epochs, batch_size=8, lr=3e-4)
    m = evaluate_router(full_ft_model, tokenizer, shift_eval, device, threshold=router_cfg["eval"]["threshold"], judge=judge)
    print(f"exact_match={m['exact_match_accuracy']:.3f}  jaccard={m['jaccard_similarity']:.3f}")
    results["full_finetune"] = {
        "exact_match_accuracy": m["exact_match_accuracy"], "jaccard_similarity": m["jaccard_similarity"],
        "trainable_params": trainable, "total_params": total,
    }

    print(f"\n=== [3/3] QLoRA ({args.num_bits}-bit frozen base + rank-{args.lora_r} adapters) ===")
    qlora_model = load_base_model()
    apply_qlora(qlora_model, target_substrings=("head", "linear1", "linear2"),
                num_bits=args.num_bits, r=args.lora_r, alpha=args.lora_r * 2)
    trainable_q, total_q = count_trainable_params(qlora_model)
    print(f"trainable params: {trainable_q:,} / {total_q:,} ({100*trainable_q/total_q:.2f}%)")
    with timer("QLoRA fine-tune"):
        finetune(qlora_model, tokenizer, list(shift_train), device, epochs=args.epochs, batch_size=8, lr=1e-3)
    m = evaluate_router(qlora_model, tokenizer, shift_eval, device, threshold=router_cfg["eval"]["threshold"], judge=judge)
    print(f"exact_match={m['exact_match_accuracy']:.3f}  jaccard={m['jaccard_similarity']:.3f}")
    results["qlora"] = {
        "exact_match_accuracy": m["exact_match_accuracy"], "jaccard_similarity": m["jaccard_similarity"],
        "trainable_params": trainable_q, "total_params": total_q, "num_bits": args.num_bits, "lora_r": args.lora_r,
    }

    print("\n=== Quantization error (base router's head weight, standalone) ===")
    base_model = load_base_model()
    head_weight = base_model.head[-1].weight.data
    qerr = {b: quantization_error(head_weight, num_bits=b) for b in (8, 4)}
    for b, e in qerr.items():
        print(f"{b}-bit: relative L2 error={e['relative_l2_error']:.4f}  compression={e['compression_ratio']:.1f}x")
    results["quantization_error"] = qerr

    print("\n=== Summary ===")
    print(f"{'config':16s} {'exact_match':>12s} {'jaccard':>10s} {'trainable %':>12s}")
    for name in ("zero_shot", "full_finetune", "qlora"):
        r = results[name]
        pct = f"{100*r['trainable_params']/r['total_params']:.2f}%" if "trainable_params" in r else "n/a"
        print(f"{name:16s} {r['exact_match_accuracy']:12.3f} {r['jaccard_similarity']:10.3f} {pct:>12s}")

    os.makedirs("results", exist_ok=True)
    save_json(results, "results/qlora_finetune.json")
    print("\nSaved results to results/qlora_finetune.json")


if __name__ == "__main__":
    main()
