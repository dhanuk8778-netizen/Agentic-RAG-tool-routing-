"""
Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023), greedy
variant: a small, fast "draft" model proposes `gamma` tokens autoregressively;
the larger "target" model then verifies all `gamma` positions in a SINGLE
forward pass (since verification, unlike drafting, doesn't need to be
sequential -- the target model scores every proposed position in parallel
given the draft's proposed continuation). Accepted tokens are every prefix
position where the target's own greedy argmax matches what the draft
proposed; at the first mismatch (or after all gamma accepted), the target's
own greedy token is appended and drafting resumes from there.

This greedy/argmax variant (rather than the original paper's stochastic
rejection sampling for temperature>0 sampling) is deliberately simple to
state a clean correctness property for: the accepted output is BYTE-FOR-BYTE
identical to running the target model alone with greedy decoding --
speculative decoding changes only how many forward passes it costs to get
that output, never what the output is. This is verified directly in
tests/test_speculative_decoding.py.
"""
from __future__ import annotations

import time

import torch

from src.serving.tiny_transformer import TinyGPT


@torch.no_grad()
def greedy_decode_baseline(model: TinyGPT, prompt_ids: list[int], max_new_tokens: int, device) -> list[int]:
    """Reference path: target model alone, one token at a time, using its
    own KV cache (still the standard single-model decode -- the comparison
    point for speculative decoding's speedup, not a strawman)."""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, past_kv = model(input_ids, use_cache=True)
    next_id = int(logits[:, -1, :].argmax(dim=-1).item())
    generated = [next_id]
    cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens - 1):
        logits, past_kv = model(cur, past_kv=past_kv, use_cache=True)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        generated.append(next_id)
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return generated


def _slice_kv(kv: list, length: int) -> list:
    """Truncate every layer's cached (k, v) to the first `length` time steps
    -- used to discard a rejected draft continuation's cache entries while
    keeping the (still-valid) accepted-prefix entries, avoiding a full
    context replay."""
    return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in kv]


@torch.no_grad()
def speculative_decode(draft: TinyGPT, target: TinyGPT, prompt_ids: list[int], max_new_tokens: int, gamma: int, device):
    """Returns (generated_ids, stats) where stats tracks how many draft
    proposals were accepted vs. rejected -- the efficiency signal (higher
    acceptance rate = closer to gamma-x fewer target forward passes).

    Correctness depends on getting logit/position alignment exactly right:
    the model's output at the position of token y_k predicts y_{k+1}, NOT
    y_k itself -- so the prediction used to accept/reject y_k must come
    from the position *before* y_k (either the previous round's trailing
    logits, or the prefill's last-position logits for y_1). Both models'
    "next-token logits" are threaded through the loop explicitly for this
    reason, rather than re-deriving them by re-feeding an already-cached
    token (which would silently duplicate that position).
    """
    generated: list[int] = []
    context_len = len(prompt_ids)

    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    draft_logits0, draft_kv = draft(inp, use_cache=True)
    target_logits0, target_kv = target(inp, use_cache=True)
    draft_next = draft_logits0[:, -1, :]
    target_next = target_logits0[:, -1, :]

    accepted_total, proposed_total, rounds = 0, 0, 0

    while len(generated) < max_new_tokens:
        rounds += 1
        step_gamma = min(gamma, max_new_tokens - len(generated))

        # 1) draft proposes step_gamma tokens autoregressively, using its
        #    own running `draft_next` logits (no cache re-feeding).
        proposals = []
        d_kv = draft_kv
        cur_logits = draft_next
        for _ in range(step_gamma):
            tok = int(cur_logits.argmax(dim=-1).item())
            proposals.append(tok)
            cur_input = torch.tensor([[tok]], dtype=torch.long, device=device)
            step_logits, d_kv = draft(cur_input, past_kv=d_kv, use_cache=True)
            cur_logits = step_logits[:, -1, :]
        proposed_total += len(proposals)
        draft_next_after_round = cur_logits  # prediction for token after the last proposal

        # 2) target verifies all proposed positions in ONE forward pass.
        verify_input = torch.tensor([proposals], dtype=torch.long, device=device)
        target_logits, verify_kv = target(verify_input, past_kv=target_kv, use_cache=True)

        # greedy_targets[i] is what the target would have picked for
        # proposals[i]: for i=0 that's `target_next` (carried in from
        # before this round); for i>0 it's the target's own output at
        # position i-1 of this round's verify pass (predicting the token
        # after proposals[i-1], i.e. proposals[i]'s slot).
        greedy_targets = [int(target_next.argmax(dim=-1).item())]
        for i in range(step_gamma - 1):
            greedy_targets.append(int(target_logits[:, i, :].argmax(dim=-1).item()))

        # 3) accept the longest matching prefix
        accepted = 0
        for p_tok, t_tok in zip(proposals, greedy_targets):
            if p_tok == t_tok:
                accepted += 1
            else:
                break
        accepted_total += accepted
        generated.extend(proposals[:accepted])

        if accepted < len(proposals):
            fallback_tok = greedy_targets[accepted]
            generated.append(fallback_tok)
            fb_input = torch.tensor([[fallback_tok]], dtype=torch.long, device=device)

            target_kv = _slice_kv(verify_kv, context_len + accepted)
            fb_target_logits, target_kv = target(fb_input, past_kv=target_kv, use_cache=True)
            target_next = fb_target_logits[:, -1, :]

            draft_kv = _slice_kv(d_kv, context_len + accepted)
            fb_draft_logits, draft_kv = draft(fb_input, past_kv=draft_kv, use_cache=True)
            draft_next = fb_draft_logits[:, -1, :]

            context_len += accepted + 1
        else:
            target_kv = verify_kv
            target_next = target_logits[:, step_gamma - 1, :]
            draft_kv = d_kv
            draft_next = draft_next_after_round
            context_len += step_gamma

        if len(generated) >= max_new_tokens:
            break

    generated = generated[:max_new_tokens]
    stats = {
        "rounds": rounds, "accepted_total": accepted_total, "proposed_total": proposed_total,
        "acceptance_rate": accepted_total / max(proposed_total, 1),
        "target_forward_passes": rounds + 1,  # +1 for the initial prompt prefill
        "tokens_per_target_pass": len(generated) / max(rounds + 1, 1),
    }
    return generated, stats
def benchmark_speculative(draft: TinyGPT, target: TinyGPT, prompts: list[list[int]], max_new_tokens: int, gamma: int, device):
    baseline_times, spec_times = [], []
    acceptance_rates = []
    mismatches = 0
    for prompt_ids in prompts:
        t0 = time.perf_counter()
        baseline_out = greedy_decode_baseline(target, prompt_ids, max_new_tokens, device)
        baseline_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        spec_out, stats = speculative_decode(draft, target, prompt_ids, max_new_tokens, gamma, device)
        spec_times.append(time.perf_counter() - t0)
        acceptance_rates.append(stats["acceptance_rate"])

        if spec_out != baseline_out:
            mismatches += 1

    mean_baseline = sum(baseline_times) / len(baseline_times)
    mean_spec = sum(spec_times) / len(spec_times)
    return {
        "num_prompts": len(prompts),
        "mismatches": mismatches,  # must be 0 -- correctness guarantee
        "mean_baseline_latency_s": mean_baseline,
        "mean_speculative_latency_s": mean_spec,
        "speedup_x": mean_baseline / mean_spec if mean_spec > 0 else float("nan"),
        "mean_acceptance_rate": sum(acceptance_rates) / len(acceptance_rates),
        "gamma": gamma,
    }
