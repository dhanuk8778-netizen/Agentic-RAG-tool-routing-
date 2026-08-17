"""
Train the semantic router (multi-label tool classifier) on synthetic
multi-intent queries, then evaluate tool-selection quality both by exact
multi-label match and via LLM-as-a-judge (real API if ANTHROPIC_API_KEY is
set, deterministic rule-based fallback otherwise).

Usage:
    python -m src.train_router --config configs/router_config.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.data.router_queries import generate_router_queries
from src.data.tokenizer import SimpleTokenizer
from src.router.llm_judge import get_judge
from src.router.semantic_router import SemanticRouter
from src.router.tools import TOOL_NAMES
from src.utils import save_json, set_seed, timer


def labels_to_multihot(tool_labels: list[str]) -> list[float]:
    return [1.0 if t in tool_labels else 0.0 for t in TOOL_NAMES]


def train_router(model, tokenizer, examples, device, epochs, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(examples) - batch_size + 1, batch_size):
            batch = examples[i : i + batch_size]
            input_ids = torch.tensor([tokenizer.encode(ex.text) for ex in batch], dtype=torch.long, device=device)
            labels = torch.tensor([labels_to_multihot(ex.tool_labels) for ex in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"[router] epoch {epoch:02d} | loss {avg:.4f}")
        history.append({"epoch": epoch, "loss": avg})
    return history


@torch.no_grad()
def evaluate_router(model, tokenizer, examples, device, threshold: float, judge, judge_sample_size: int | None = None):
    model.eval()
    exact_match = 0
    jaccard_sum = 0.0
    judge_correct = 0
    judge_n = 0

    judge_indices = set(range(len(examples)))
    if judge_sample_size is not None and judge_sample_size < len(examples):
        judge_indices = set(random.Random(0).sample(range(len(examples)), judge_sample_size))

    per_example = []
    for i, ex in enumerate(examples):
        input_ids = torch.tensor([tokenizer.encode(ex.text)], dtype=torch.long, device=device)
        predicted = model.route(input_ids, threshold=threshold)[0]
        ref = set(ex.tool_labels)
        pred = set(predicted)
        exact_match += int(pred == ref)
        union = pred | ref
        jaccard_sum += (len(pred & ref) / len(union)) if union else 1.0

        judged = None
        if i in judge_indices:
            verdict = judge.judge(ex.text, predicted, ex.tool_labels)
            judge_correct += int(verdict.correct)
            judge_n += 1
            judged = {"correct": verdict.correct, "rationale": verdict.rationale}

        per_example.append({"query": ex.text, "predicted": predicted, "reference": ex.tool_labels, "judge": judged})

    n = len(examples)
    return {
        "exact_match_accuracy": exact_match / n,
        "jaccard_similarity": jaccard_sum / n,
        "llm_judge_accuracy": (judge_correct / judge_n) if judge_n else None,
        "llm_judge_backend": judge.name,
        "num_examples": n,
        "num_judged": judge_n,
        "per_example_sample": per_example[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/router_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cpu")

    with timer("query generation"):
        all_examples = generate_router_queries(
            num_queries=cfg["data"]["num_queries"],
            multi_intent_fraction=cfg["data"]["multi_intent_fraction"],
            seed=cfg["seed"],
        )
    split = int(len(all_examples) * cfg["data"]["train_fraction"])
    train_examples, eval_examples = all_examples[:split], all_examples[split:]
    print(f"train={len(train_examples)} eval={len(eval_examples)} "
          f"multi_intent_frac~{sum(len(e.tool_labels) > 1 for e in all_examples) / len(all_examples):.2f}")

    tokenizer = SimpleTokenizer.build([e.text for e in all_examples], vocab_size=cfg["tokenizer"]["vocab_size"], max_len=cfg["tokenizer"]["max_len"])
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    tokenizer.save(os.path.join(cfg["output"]["dir"], "router_tokenizer.json"))

    model = SemanticRouter(tokenizer_vocab_size=tokenizer.vocab_size, **cfg["model"]).to(device)

    with timer("router training"):
        history = train_router(model, tokenizer, train_examples, device, epochs=cfg["train"]["epochs"], batch_size=cfg["train"]["batch_size"], lr=cfg["train"]["lr"])
    torch.save(model.state_dict(), os.path.join(cfg["output"]["dir"], "semantic_router.pt"))

    judge = get_judge()
    with timer("router evaluation"):
        metrics = evaluate_router(
            model, tokenizer, eval_examples, device,
            threshold=cfg["eval"]["threshold"], judge=judge, judge_sample_size=cfg["eval"].get("judge_sample_size"),
        )

    print(f"\nExact multi-label match accuracy: {metrics['exact_match_accuracy']:.3f}")
    print(f"Jaccard similarity (partial credit): {metrics['jaccard_similarity']:.3f}")
    if metrics["llm_judge_accuracy"] is not None:
        print(f"LLM-as-a-judge tool-selection accuracy ({metrics['llm_judge_backend']}): {metrics['llm_judge_accuracy']:.3f} "
              f"(n={metrics['num_judged']})")

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    save_json({"config": cfg, "history": history, "metrics": metrics}, os.path.join(cfg["output"]["results_dir"], "router_eval.json"))
    print(f"\nSaved results to {cfg['output']['results_dir']}/router_eval.json")


if __name__ == "__main__":
    main()
