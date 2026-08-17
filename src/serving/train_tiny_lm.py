"""
Trains two TinyGPT checkpoints on the project's own synthetic corpus text
(next-token prediction, character-level): a "target" model and a smaller,
faster "draft" model -- the pair speculative decoding needs. Also used
standalone by the KV-cache benchmark.

Usage:
    python -m src.serving.train_tiny_lm --config configs/serving_config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.data.corpus import generate_corpus
from src.serving.char_tokenizer import CharTokenizer
from src.serving.tiny_transformer import TinyGPT
from src.utils import save_json, set_seed, timer


def build_training_text(num_docs: int, seed: int) -> str:
    docs = generate_corpus(num_docs=num_docs, seed=seed)
    return "\n".join(d.text for d in docs)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def train_lm(model, data, device, steps, block_size, batch_size, lr, log_every=200):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []
    for step in range(1, steps + 1):
        x, y = get_batch(data, block_size, batch_size, device)
        logits, _ = model(x, use_cache=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps:
            print(f"  step {step:5d}/{steps} | loss {loss.item():.4f}")
            history.append({"step": step, "loss": loss.item()})
    return history


def train_draft_via_distillation(draft, target, data, device, steps, block_size, batch_size, lr, log_every=200):
    """Train the draft model to match the TARGET model's own predictions
    (hard-label distillation on the target's argmax) rather than the raw
    training data. This is what real speculative-decoding draft models are
    trained for: independent training on the same corpus gives a
    reasonable language model, but nothing directly optimizes it to *agree
    with the target*, which is the only thing that determines speculative
    decoding's acceptance rate (and therefore its speedup) -- see README
    "Why the draft model is distilled, not just independently trained".
    """
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(draft.parameters(), lr=lr)
    draft.train()
    history = []
    for step in range(1, steps + 1):
        x, _ = get_batch(data, block_size, batch_size, device)
        with torch.no_grad():
            target_logits, _ = target(x, use_cache=False)
            target_labels = target_logits.argmax(dim=-1)  # (B, T) hard pseudo-labels
        draft_logits, _ = draft(x, use_cache=False)
        loss = F.cross_entropy(draft_logits.reshape(-1, draft_logits.size(-1)), target_labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps:
            agreement = (draft_logits.argmax(dim=-1) == target_labels).float().mean().item()
            print(f"  step {step:5d}/{steps} | distill_loss {loss.item():.4f} | draft-target agreement {agreement:.3f}")
            history.append({"step": step, "loss": loss.item(), "agreement": agreement})
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/serving_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device("cpu")

    with timer("corpus text build"):
        text = build_training_text(cfg["lm_data"]["num_docs"], cfg["seed"])
    print(f"training text length: {len(text):,} chars")

    tok = CharTokenizer()
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    tok.save(os.path.join(cfg["output"]["dir"], "char_tokenizer.json"))
    data = torch.tensor(tok.encode(text, add_bos=False), dtype=torch.long)

    print("\n=== Training TARGET model ===")
    target = TinyGPT(vocab_size=tok.vocab_size, **cfg["target_model"])
    print(f"target params: {target.num_parameters():,}")
    with timer("target training"):
        target_history = train_lm(target, data, device, steps=cfg["train"]["target_steps"],
                                   block_size=cfg["train"]["block_size"], batch_size=cfg["train"]["batch_size"], lr=cfg["train"]["lr"])
    torch.save(target.state_dict(), os.path.join(cfg["output"]["dir"], "tiny_gpt_target.pt"))

    print("\n=== Training DRAFT model (distilled from target, for speculative decoding) ===")
    draft = TinyGPT(vocab_size=tok.vocab_size, **cfg["draft_model"])
    print(f"draft params: {draft.num_parameters():,}")
    with timer("draft distillation"):
        draft_history = train_draft_via_distillation(draft, target, data, device, steps=cfg["train"]["draft_steps"],
                                                       block_size=cfg["train"]["block_size"], batch_size=cfg["train"]["batch_size"], lr=cfg["train"]["lr"])
    torch.save(draft.state_dict(), os.path.join(cfg["output"]["dir"], "tiny_gpt_draft.pt"))

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    save_json(
        {"config": cfg, "target_history": target_history, "draft_history": draft_history,
         "target_params": target.num_parameters(), "draft_params": draft.num_parameters()},
        os.path.join(cfg["output"]["results_dir"], "tiny_lm_train.json"),
    )
    print(f"\nSaved checkpoints to {cfg['output']['dir']}/ and log to {cfg['output']['results_dir']}/tiny_lm_train.json")


if __name__ == "__main__":
    main()
