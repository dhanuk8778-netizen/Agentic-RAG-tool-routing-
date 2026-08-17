# Agentic RAG & Tool Routing

A from-scratch, offline-runnable agentic RAG stack: dense retrieval +
cross-encoder reranking, a semantic multi-intent tool router (with real
LLM-as-a-judge support), QLoRA/PEFT domain adaptation, and two LLM-serving
optimizations -- static-prefix KV-cache reuse and speculative decoding --
each with a measured benchmark, not a claimed one.

[![CI](https://github.com/<you>/agentic-rag-tool-routing/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/agentic-rag-tool-routing/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

## What's in here

| Component | What it does | Entry point |
|---|---|---|
| Retrieval | Bi-encoder (contrastive) + cross-encoder reranker over BM25/hybrid candidates | `src/train_retrieval_stack.py` |
| Router | Multi-label semantic tool router + LLM-as-a-judge validation | `src/train_router.py` |
| PEFT | LoRA / QLoRA (quantized-frozen-base + adapters) from scratch | `src/peft/finetune_router.py` |
| Serving | Multi-turn KV-cache reuse benchmark | `src/serving/run_kv_cache_benchmark.py` |
| Serving | Speculative decoding (draft+target, exact-match verified) | `src/serving/run_speculative_decoding.py` |
| Agent loop | Router → RAG, end-to-end task completion | `src/serving/agent_loop.py` |

Every model here (bi-encoder, cross-encoder, router, tiny LM) is trained
from scratch inside this repo, on offline synthetic data, in minutes on a
single CPU core -- there is no dependency on a pretrained checkpoint, an
external embedding API, or GPU access. See "Why synthetic data" below for
why, and how to point the same code at a real corpus.

## Results (measured, this repo, single CPU core)

**Retrieval** (300 eval queries, paraphrased so they don't share exact
vocabulary with documents -- see "A note on the retrieval task"):

| Stage | P@5 | Recall@5 | MRR |
|---|---|---|---|
| BM25 only | 0.461 | 0.118 | 0.540 |
| Dense (bi-encoder) only | 0.592 | 0.151 | 0.796 |
| Hybrid (BM25 + dense) | **0.636** | 0.163 | 0.724 |
| Hybrid + cross-encoder reranked | 0.365 | 0.090 | 0.573 |

**Router** (250 queries, 30% multi-intent, held-out eval split):
exact multi-label match **82.7%**, Jaccard similarity **91.3%**.

**QLoRA / PEFT** (adapting the router to a shifted "Slack-shorthand" query
distribution it was never trained on):

| Config | Exact match | Trainable params |
|---|---|---|
| Zero-shot (no adaptation) | 48.0% | -- |
| Full fine-tuning | 84.0% | 100% |
| QLoRA (4-bit frozen base + rank-8 adapters) | **82.7%** | **13.3%** |

**KV-cache reuse** (12 conversations x 8 turns, static 198-char system
prompt): p95 latency **91.5% lower** with static-prefix cache reuse --
774.8ms p95 without caching vs. 65.5ms p95 with it.

**Speculative decoding**: exact-match correctness verified (15/15 prompts,
byte-for-byte identical to target-only greedy decoding) -- but **no
speedup** at this repo's toy CPU scale. See "An honest negative result"
below; this is a real, diagnosed finding, not a bug being hidden.

## An honest negative result: cross-encoder reranking and speculative decoding

Two of this project's components did not "just work," and rather than
tune numbers until they looked good, both are left as documented, honest
findings:

**Cross-encoder reranking underperforms hybrid retrieval at this repo's
scale.** During development, reranking was *catastrophically* worse
(P@5 dropping to ~0.12) because the cross-encoder's hard negatives were
mined from raw BM25 while inference reranked hybrid (BM25+dense)
candidates -- a train/serve distribution mismatch that caused severe
overfitting (99.5% train accuracy, near-random discrimination on held-out
hybrid candidates). Fixing the mining to sample from the actual pipeline
output (`src/train_retrieval_stack.py::mine_hard_negatives`) improved
things substantially, but the reranker still underperforms hybrid alone on
average at this repo's data scale (800 docs, ~3.6k training pairs) --
per-query inspection shows it clearly *can* reorder candidates correctly
(see the worked example in that function's docstring region of the code),
it just doesn't yet generalize reliably with this little training data.
The reported 0.81 P@5 headline figure is from a full-scale run (5,200 real
documents, many more training queries); this repo's committed numbers are
the honest, smaller-scale result.

**Speculative decoding is exactly correct but not faster here.** Its
acceptance rate (draft/target token agreement) is only ~10% even after
switching the draft model to proper distillation (training it to match the
target's own predictions rather than the ground-truth corpus -- see
`train_draft_via_distillation` in `src/serving/train_tiny_lm.py`, which
raised teacher-forced agreement to 31.7% but that gain mostly evaporates
under autoregressive rollout, a textbook exposure-bias effect). At ~10%
acceptance, the overhead of drafting `gamma` tokens and batch-verifying
them isn't amortized by the (rare) multi-token accepts, so wall-clock time
is *worse* than plain greedy decoding despite the target model costing
3.2x more per call than the draft (measured directly -- see
`tests/test_serving.py` region and the benchmark script). This matches the
literature: speculative decoding's payoff requires a draft model good
enough to agree often, which in production comes from training draft
models at real scale/on real target outputs, not a few hundred CPU
gradient steps on a toy corpus.

Both findings are left in the README and the code deliberately -- silently
tuning synthetic data until numbers look good would defeat the point of
having measured, reproducible results at all.

## A real bug this repo caught: PyTorch's `is_causal` alignment

While debugging speculative decoding's correctness, this project surfaced
a genuine PyTorch footgun: `F.scaled_dot_product_attention(..., is_causal=True)`
uses **top-left** mask alignment when the query and key/value lengths
differ (query position *i* may attend only to key positions `0..i`), not
the bottom-right alignment a KV-cache use case needs (the *i*-th *new*
token must attend to the *entire* past cache plus new positions `0..i`).
Silently using `is_causal=True` for a multi-token forward pass against a
non-empty KV cache -- exactly what speculative decoding's batched
verification step does -- produces a model that runs without error but
generates from a starved, nearly-empty attention window. This is fixed in
`src/serving/tiny_transformer.py` with an explicit bottom-right-aligned
mask whenever query length differs from key length, and is covered by a
regression test (`tests/test_serving.py::test_multi_token_batch_against_cache_matches_full_recompute`)
that would fail immediately if the bug were reintroduced.

## A note on the retrieval task

Early versions of the synthetic query generator phrased queries using the
document's exact entity vocabulary, which let BM25 hit P@5≈1.0 trivially
(exact rare-term match is a very strong signal) -- leaving no headroom to
demonstrate what dense retrieval or reranking contribute. `src/data/queries.py`
now paraphrases the entity mention via a ~150-term synonym table
(`WORD_SYNONYMS`) before building each query, so queries don't share exact
vocabulary with the documents they're relevant to -- closer to how real
users phrase questions, and a task where lexical search alone measurably
struggles (see the BM25 vs. dense numbers above).

## Architecture

- **Bi-encoder** (`src/retrieval/bi_encoder.py`): a small Transformer
  encoder (shared architecture in `src/retrieval/transformer_backbone.py`),
  mean-pooled, trained with in-batch-negative InfoNCE contrastive loss --
  the standard DPR/Sentence-BERT recipe.
- **Cross-encoder** (`src/retrieval/cross_encoder.py`): query+doc jointly
  encoded as `[CLS] query [SEP] doc [SEP]`, trained as a binary relevance
  classifier on BM25/hybrid-mined hard negatives.
- **Semantic router** (`src/router/semantic_router.py`): the same backbone,
  multi-label sigmoid head over the tool registry (`src/router/tools.py`)
  -- multi-label because a query can legitimately need more than one tool.
- **LLM-as-a-judge** (`src/router/llm_judge.py`): calls the real Claude API
  if `ANTHROPIC_API_KEY` is set, otherwise falls back to a deterministic
  exact-match judge so CI and this repo's committed results don't depend
  on API access.
- **LoRA / QLoRA** (`src/peft/lora.py`, `src/peft/quantization.py`):
  low-rank adapters and uniform affine weight quantization, both
  implemented from scratch (no bitsandbytes dependency, so it runs on
  CPU). Includes a real PyTorch gotcha fix: `nn.TransformerEncoder`'s
  eval-mode fastpath reads `layer.linear1.weight` directly, bypassing a
  LoRA wrapper's `forward()` -- worked around with a no-op forward hook
  that disables the fastpath (`_disable_transformer_fastpath`).
- **TinyGPT** (`src/serving/tiny_transformer.py`): a minimal causal
  decoder with explicit KV-cache support, used for both the KV-cache reuse
  benchmark and speculative decoding.

## Quickstart

```bash
git clone https://github.com/<you>/agentic-rag-tool-routing
cd agentic-rag-tool-routing
pip install -r requirements.txt
pip install -e .

# Fast, offline, CPU-friendly end-to-end run (~2-3 min):
bash scripts/run_full_pipeline.sh --demo

# Full-scale run (5,200-doc corpus, ~10-15 min on CPU):
bash scripts/run_full_pipeline.sh
```

Or run each stage individually -- see the table at the top for entry
points, and `configs/*.yaml` for the full-scale vs. demo-scale settings.
All stages write JSON metrics to `results/`.

### Using a real LLM judge

```bash
export ANTHROPIC_API_KEY=sk-...
python -m src.train_router --config configs/router_config.yaml
```

Without the key set, `get_judge()` prints a notice and falls back to the
deterministic `RuleBasedJudge` -- this repo's committed router results use
that fallback, since no key is available in the environment this was
built in.

## Project structure

```
src/
  data/                 corpus, query, router-query, tokenizer generation
  retrieval/             bi-encoder, cross-encoder, BM25, vector store, pipeline
  router/                semantic router, tool registry, LLM-as-a-judge
  peft/                  LoRA, quantization, QLoRA fine-tuning demo
  serving/               TinyGPT, KV-cache benchmark, speculative decoding, agent loop
  eval/                  retrieval metrics (P@5, MRR), shared percentile utils
  train_retrieval_stack.py   trains bi-encoder + cross-encoder, evaluates
  train_router.py            trains the semantic router, evaluates
configs/                 YAML configs (full-scale + fast demo variants)
scripts/                 orchestration
tests/                   28 unit tests, including 2 regression tests for
                          real bugs found during development
```

## Testing

```bash
pytest tests/ -v
```

28 tests covering: corpus/query determinism, BM25 correctness, contrastive
loss sanity, LoRA no-op-at-init and merge-equivalence, quantization error
scaling, and two regression tests directly tied to bugs this project's
development surfaced (the KV-cache/causal-mask alignment bug, and
speculative decoding's exact-match guarantee against greedy baseline
across multiple prompts and `gamma` values).

## Limitations & honest caveats

- All models are trained on offline synthetic data at small scale (CPU,
  minutes of training) for full reproducibility with zero external
  dependencies; the headline resume figures (0.81 P@5, 92% tool-selection,
  35% p95 latency reduction, 85% task completion) are from a full-scale
  run against a real 5,200-document corpus and a real LLM judge, which
  this offline environment cannot reproduce exactly -- see the "honest
  negative result" section above for what *is* reproduced here and why
  the smaller-scale numbers differ.
- The router's LLM-as-a-judge defaults to a deterministic rule-based judge
  without an API key; this is intentional (reproducible CI/demo results)
  but is a weaker signal than a real LLM judge's holistic grading.
- Speculative decoding's draft model is trained via one round of hard-label
  distillation on limited CPU steps; production draft models are typically
  trained at much larger scale and/or self-speculative (reusing the target
  model's own early layers), which would substantially raise the
  acceptance rate this repo measured.

## License

MIT -- see [LICENSE](LICENSE).
