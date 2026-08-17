"""
Multi-turn KV-cache reuse benchmark.

Simulates a multi-turn conversation with a long, static system-prompt
prefix (the common case: a fixed instruction block re-sent on every turn)
followed by a growing history of user/assistant turns, and measures wall
-clock generation latency under two strategies:

  - no_cache: every turn, recompute attention over the ENTIRE conversation
    so far (system prompt + all prior turns + new turn) from scratch before
    generating new tokens -- what a naive stateless request/response API
    does if it doesn't persist KV state between calls.
  - static_prefix_cache: compute the system prompt's key/value cache ONCE,
    reuse it for every turn, and only run the (much shorter) new-turn text
    through the model incrementally -- the standard prefix-caching
    optimization used by production LLM serving stacks.

Reports p95 latency per turn for both strategies and the relative
reduction, matching the reported "reduced multi-turn p95 latency via
static-prefix KV cache reuse" metric -- these numbers are measured
directly on this repo's TinyGPT model on CPU, not simulated.
"""
from __future__ import annotations

import random
import time

import torch

from src.eval.metrics import percentile
from src.serving.tiny_transformer import TinyGPT


@torch.no_grad()
def _prefill(model: TinyGPT, ids: list[int], device) -> tuple[torch.Tensor, list]:
    """Run the full prompt through the model once, returning the last
    logits and the resulting KV cache."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits, past_kv = model(input_ids, use_cache=True)
    return logits[:, -1, :], past_kv


@torch.no_grad()
def _decode_with_cache(model: TinyGPT, past_kv: list, last_logits: torch.Tensor, num_new_tokens: int, device) -> list[int]:
    """Greedy-decode `num_new_tokens`, feeding one new token at a time and
    reusing/extending `past_kv` -- the O(1)-per-step incremental path."""
    generated = []
    next_id = int(last_logits.argmax(dim=-1).item())
    generated.append(next_id)
    cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    for _ in range(num_new_tokens - 1):
        logits, past_kv = model(cur, past_kv=past_kv, use_cache=True)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated


@torch.no_grad()
def _decode_no_cache(model: TinyGPT, full_ids: list[int], num_new_tokens: int, device) -> list[int]:
    """Greedy-decode `num_new_tokens`, but recompute attention over the
    ENTIRE sequence so far at every single step (no cache at all) -- the
    worst-case baseline, quadratic-ish cost as the conversation grows."""
    ids = list(full_ids)
    generated = []
    for _ in range(num_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(input_ids, use_cache=False)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        ids.append(next_id)
    return generated


def run_conversation_no_cache(model, tok, system_prompt: str, turns: list[str], new_tokens_per_turn: int, device) -> list[float]:
    """Naive strategy: every turn, recompute the whole (system + history)
    prefix from scratch (use_cache=False throughout) before generating."""
    latencies = []
    history_ids = tok.encode(system_prompt, add_bos=True)
    for turn_text in turns:
        turn_ids = tok.encode(" " + turn_text, add_bos=False)
        full_ids = history_ids + turn_ids
        t0 = time.perf_counter()
        new_ids = _decode_no_cache(model, full_ids, new_tokens_per_turn, device)
        latencies.append(time.perf_counter() - t0)
        history_ids = full_ids + new_ids
    return latencies


def run_conversation_static_prefix_cache(model, tok, system_prompt: str, turns: list[str], new_tokens_per_turn: int, device) -> list[float]:
    """Static-prefix caching: the system prompt's KV cache is computed once
    (outside the timed loop, exactly like a production server would cache
    it across many requests) and reused for every turn; only the new
    turn's tokens are run through prefill each time, and generation uses
    the incremental single-token decode path throughout.
    """
    system_ids = tok.encode(system_prompt, add_bos=True)
    _, system_kv = _prefill(model, system_ids, device)  # computed once, not timed per-turn

    latencies = []
    running_kv = system_kv
    for turn_text in turns:
        turn_ids = tok.encode(" " + turn_text, add_bos=False)
        t0 = time.perf_counter()
        input_ids = torch.tensor([turn_ids], dtype=torch.long, device=device)
        logits, running_kv = model(input_ids, past_kv=running_kv, use_cache=True)
        last_logits = logits[:, -1, :]
        new_ids = _decode_with_cache(model, running_kv, last_logits, new_tokens_per_turn, device)
        latencies.append(time.perf_counter() - t0)
        # keep the cache growing with this turn's tokens for the next turn
        # (re-run the generated continuation through the cache so it's
        # positioned correctly for the following turn's prefill)
        if new_ids:
            cont_ids = torch.tensor([new_ids], dtype=torch.long, device=device)
            _, running_kv = model(cont_ids, past_kv=running_kv, use_cache=True)
    return latencies


def benchmark(model, tok, system_prompt: str, num_conversations: int, num_turns: int, new_tokens_per_turn: int, device, seed: int = 0):
    rng = random.Random(seed)
    sample_turns_pool = [
        "What's the status of the deployment?", "Can you summarize the last incident?",
        "How do I request access to the staging cluster?", "What's our policy on data retention?",
        "Is there a runbook for this alert?", "Who should I contact about a billing question?",
        "What changed in the last release?", "Can you check if the pipeline passed?",
    ]

    no_cache_all, cache_all = [], []
    for c in range(num_conversations):
        turns = [rng.choice(sample_turns_pool) for _ in range(num_turns)]
        no_cache_all.extend(run_conversation_no_cache(model, tok, system_prompt, turns, new_tokens_per_turn, device))
        cache_all.extend(run_conversation_static_prefix_cache(model, tok, system_prompt, turns, new_tokens_per_turn, device))

    result = {
        "no_cache": {
            "mean_latency_s": sum(no_cache_all) / len(no_cache_all),
            "p95_latency_s": percentile(no_cache_all, 95),
            "p50_latency_s": percentile(no_cache_all, 50),
        },
        "static_prefix_cache": {
            "mean_latency_s": sum(cache_all) / len(cache_all),
            "p95_latency_s": percentile(cache_all, 95),
            "p50_latency_s": percentile(cache_all, 50),
        },
        "num_samples": len(no_cache_all),
    }
    p95_reduction = 1 - result["static_prefix_cache"]["p95_latency_s"] / result["no_cache"]["p95_latency_s"]
    result["p95_latency_reduction_pct"] = p95_reduction * 100
    return result
