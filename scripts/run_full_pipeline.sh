#!/usr/bin/env bash
# Runs the entire project pipeline end-to-end:
#   1. Train the retrieval stack (bi-encoder + cross-encoder), evaluate P@5
#   2. Train the semantic router, evaluate tool-selection accuracy
#   3. QLoRA/PEFT domain-adaptation demo on the router
#   4. Train the tiny LM (target + distilled draft)
#   5. KV-cache reuse benchmark (multi-turn latency)
#   6. Speculative decoding benchmark (correctness + speedup)
#   7. Agent loop: router + RAG tied together, end-to-end task completion
#
# Usage:
#   bash scripts/run_full_pipeline.sh            # full-scale configs
#   bash scripts/run_full_pipeline.sh --demo      # fast demo-scale configs
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

if [[ "${1:-}" == "--demo" ]]; then
  RETRIEVAL_CFG=configs/demo_retrieval_config.yaml
else
  RETRIEVAL_CFG=configs/retrieval_config.yaml
fi

echo "=== [1/7] Training retrieval stack ==="
python -m src.train_retrieval_stack --config "$RETRIEVAL_CFG"

echo "=== [2/7] Training semantic router ==="
python -m src.train_router --config configs/router_config.yaml

echo "=== [3/7] QLoRA/PEFT domain-adaptation demo ==="
python -m src.peft.finetune_router --router-config configs/router_config.yaml

echo "=== [4/7] Training tiny LM (target + distilled draft) ==="
python -m src.serving.train_tiny_lm --config configs/serving_config.yaml

echo "=== [5/7] KV-cache reuse benchmark ==="
python -m src.serving.run_kv_cache_benchmark --config configs/serving_config.yaml

echo "=== [6/7] Speculative decoding benchmark ==="
python -m src.serving.run_speculative_decoding --config configs/serving_config.yaml

echo "=== [7/7] Agent loop (router + RAG, E2E task completion) ==="
python -m src.serving.agent_loop --router-config configs/router_config.yaml --retrieval-config "$RETRIEVAL_CFG"

echo "Done. See results/*.json for all metrics."
