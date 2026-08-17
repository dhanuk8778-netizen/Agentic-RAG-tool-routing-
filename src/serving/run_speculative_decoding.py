"""
Usage:
    python -m src.serving.run_speculative_decoding --config configs/serving_config.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.data.corpus import generate_corpus
from src.serving.char_tokenizer import CharTokenizer
from src.serving.speculative_decoding import benchmark_speculative
from src.serving.tiny_transformer import TinyGPT
from src.utils import save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/serving_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device("cpu")

    tok = CharTokenizer()
    target = TinyGPT(vocab_size=tok.vocab_size, **cfg["target_model"])
    target.load_state_dict(torch.load(os.path.join(cfg["output"]["dir"], "tiny_gpt_target.pt"), map_location=device))
    target.eval()
    draft = TinyGPT(vocab_size=tok.vocab_size, **cfg["draft_model"])
    draft.load_state_dict(torch.load(os.path.join(cfg["output"]["dir"], "tiny_gpt_draft.pt"), map_location=device))
    draft.eval()

    spec_cfg = cfg["speculative_decoding"]
    rng = random.Random(cfg["seed"])
    docs = generate_corpus(num_docs=50, seed=cfg["seed"] + 1)
    prompts = []
    for d in rng.sample(docs, min(spec_cfg["num_prompts"], len(docs))):
        prompt_text = d.text[:40]
        prompts.append(tok.encode(prompt_text, add_bos=True))

    print(f"Running speculative decoding benchmark: {len(prompts)} prompts, gamma={spec_cfg['gamma']}, "
          f"max_new_tokens={spec_cfg['max_new_tokens']}")
    print(f"target params={target.num_parameters():,}  draft params={draft.num_parameters():,} "
          f"({target.num_parameters()/draft.num_parameters():.1f}x larger)\n")

    result = benchmark_speculative(draft, target, prompts, max_new_tokens=spec_cfg["max_new_tokens"], gamma=spec_cfg["gamma"], device=device)

    print(f"Correctness: {result['num_prompts'] - result['mismatches']}/{result['num_prompts']} prompts "
          f"exactly matched target-only greedy decoding" + (" (PASS)" if result["mismatches"] == 0 else " (MISMATCH -- bug!)"))
    print(f"Mean acceptance rate: {result['mean_acceptance_rate']:.2f}")
    print(f"Mean baseline latency: {result['mean_baseline_latency_s']*1000:.1f}ms")
    print(f"Mean speculative latency: {result['mean_speculative_latency_s']*1000:.1f}ms")
    print(f"Speedup: {result['speedup_x']:.2f}x")

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    save_json(result, os.path.join(cfg["output"]["results_dir"], "speculative_decoding_benchmark.json"))
    print(f"\nSaved results to {cfg['output']['results_dir']}/speculative_decoding_benchmark.json")


if __name__ == "__main__":
    main()
