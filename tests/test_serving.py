import torch

from src.serving.char_tokenizer import CharTokenizer
from src.serving.speculative_decoding import greedy_decode_baseline, speculative_decode
from src.serving.tiny_transformer import TinyGPT


def _tiny_model(vocab_size: int) -> TinyGPT:
    return TinyGPT(vocab_size=vocab_size, dim=16, num_layers=2, num_heads=2, ff_dim=32, max_len=128, dropout=0.0)


def test_tiny_gpt_forward_shape():
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    ids = torch.tensor([tok.encode("hello world")])
    logits, _ = model(ids, use_cache=False)
    assert logits.shape == (1, ids.shape[1], tok.vocab_size)


def test_kv_cache_incremental_matches_full_recompute():
    """The core correctness property KV caching depends on: decoding one
    token at a time with a growing cache must produce IDENTICAL logits to
    recomputing the full sequence from scratch at each step (this is what
    the multi-token-batch causal-mask bug violated -- see
    speculative_decoding regression test below for the batched case)."""
    torch.manual_seed(0)
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()

    full_ids = tok.encode("the quick brown fox", add_bos=True)
    with torch.no_grad():
        full_logits, _ = model(torch.tensor([full_ids]), use_cache=False)

    # incremental: prefill first half, then feed the rest one token at a time
    split = len(full_ids) // 2
    with torch.no_grad():
        _, kv = model(torch.tensor([full_ids[:split]]), use_cache=True)
        incremental_logits = []
        for i in range(split, len(full_ids)):
            step_logits, kv = model(torch.tensor([[full_ids[i]]]), past_kv=kv, use_cache=True)
            incremental_logits.append(step_logits[:, -1, :])

    for i, logit in enumerate(incremental_logits):
        full_idx = split + i
        assert torch.allclose(logit, full_logits[:, full_idx, :], atol=1e-4), f"mismatch at position {full_idx}"


def test_multi_token_batch_against_cache_matches_full_recompute():
    """Regression test for the causal-mask alignment bug: feeding several
    NEW tokens at once against an existing KV cache (as speculative
    decoding's verification step does) must give the same per-position
    logits as recomputing the whole sequence from scratch -- PyTorch's
    `is_causal=True` uses top-left mask alignment when query and key
    lengths differ, which silently starves early new-token positions of
    most of the cached context. See src/serving/tiny_transformer.py's
    explicit bottom-right-aligned mask construction.
    """
    torch.manual_seed(1)
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()

    full_ids = tok.encode("pack my box with five dozen liquor jugs", add_bos=True)
    with torch.no_grad():
        full_logits, _ = model(torch.tensor([full_ids]), use_cache=False)

    split = 5
    new_chunk = full_ids[split : split + 4]
    with torch.no_grad():
        _, kv = model(torch.tensor([full_ids[:split]]), use_cache=True)
        batch_logits, _ = model(torch.tensor([new_chunk]), past_kv=kv, use_cache=True)

    for i in range(len(new_chunk)):
        full_idx = split + i
        assert torch.allclose(batch_logits[:, i, :], full_logits[:, full_idx, :], atol=1e-4), \
            f"multi-token-batch-against-cache mismatch at position {full_idx}"


def test_speculative_decoding_exact_match_with_greedy_baseline():
    """The headline correctness guarantee: speculative decoding's output
    must be byte-for-byte identical to running the target model alone."""
    torch.manual_seed(2)
    tok = CharTokenizer()
    target = _tiny_model(tok.vocab_size)
    draft = TinyGPT(vocab_size=tok.vocab_size, dim=8, num_layers=1, num_heads=2, ff_dim=16, max_len=128, dropout=0.0)
    target.eval()
    draft.eval()

    for prompt_text in ["hello", "the quick brown", "a b c d e"]:
        prompt_ids = tok.encode(prompt_text, add_bos=True)
        baseline = greedy_decode_baseline(target, prompt_ids, max_new_tokens=15, device=torch.device("cpu"))
        spec, stats = speculative_decode(draft, target, prompt_ids, max_new_tokens=15, gamma=3, device=torch.device("cpu"))
        assert spec == baseline, f"mismatch for prompt {prompt_text!r}: {spec} != {baseline}"
        assert 0.0 <= stats["acceptance_rate"] <= 1.0


def test_speculative_decoding_gamma_1_still_correct():
    """gamma=1 degenerates to a (slower) token-by-token verify loop; still
    must match exactly."""
    torch.manual_seed(3)
    tok = CharTokenizer()
    target = _tiny_model(tok.vocab_size)
    draft = TinyGPT(vocab_size=tok.vocab_size, dim=8, num_layers=1, num_heads=2, ff_dim=16, max_len=128, dropout=0.0)
    target.eval()
    draft.eval()

    prompt_ids = tok.encode("test prompt", add_bos=True)
    baseline = greedy_decode_baseline(target, prompt_ids, max_new_tokens=10, device=torch.device("cpu"))
    spec, _ = speculative_decode(draft, target, prompt_ids, max_new_tokens=10, gamma=1, device=torch.device("cpu"))
    assert spec == baseline
