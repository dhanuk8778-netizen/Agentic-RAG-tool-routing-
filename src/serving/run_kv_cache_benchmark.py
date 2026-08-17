"""
Usage:
    python -m src.serving.run_kv_cache_benchmark --config configs/serving_config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.serving.char_tokenizer import CharTokenizer
from src.serving.kv_cache import benchmark
from src.serving.tiny_transformer import TinyGPT
from src.utils import save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/serving_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="defaults to <output.dir>/tiny_gpt_target.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device("cpu")

    tok = CharTokenizer()
    model = TinyGPT(vocab_size=tok.vocab_size, **cfg["target_model"])
    ckpt_path = args.checkpoint or os.path.join(cfg["output"]["dir"], "tiny_gpt_target.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    bench_cfg = cfg["kv_cache_benchmark"]
    print(f"Running KV-cache benchmark: {bench_cfg['num_conversations']} conversations x "
          f"{bench_cfg['num_turns']} turns x {bench_cfg['new_tokens_per_turn']} new tokens/turn")
    print(f"System prompt length: {len(bench_cfg['system_prompt'])} chars\n")

    result = benchmark(
        model, tok, bench_cfg["system_prompt"],
        num_conversations=bench_cfg["num_conversations"], num_turns=bench_cfg["num_turns"],
        new_tokens_per_turn=bench_cfg["new_tokens_per_turn"], device=device, seed=cfg["seed"],
    )

    print(f"no_cache            | mean={result['no_cache']['mean_latency_s']*1000:.1f}ms  "
          f"p50={result['no_cache']['p50_latency_s']*1000:.1f}ms  p95={result['no_cache']['p95_latency_s']*1000:.1f}ms")
    print(f"static_prefix_cache | mean={result['static_prefix_cache']['mean_latency_s']*1000:.1f}ms  "
          f"p50={result['static_prefix_cache']['p50_latency_s']*1000:.1f}ms  p95={result['static_prefix_cache']['p95_latency_s']*1000:.1f}ms")
    print(f"\np95 latency reduction: {result['p95_latency_reduction_pct']:.1f}%")

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    save_json(result, os.path.join(cfg["output"]["results_dir"], "kv_cache_benchmark.json"))
    print(f"\nSaved results to {cfg['output']['results_dir']}/kv_cache_benchmark.json")


if __name__ == "__main__":
    main()
