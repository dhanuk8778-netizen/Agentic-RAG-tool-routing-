"""
Trains the full retrieval stack end-to-end on the synthetic corpus:

  1. Generate corpus + eval queries (exact relevance judgments).
  2. Build a tokenizer from the corpus vocabulary.
  3. Contrastively train the bi-encoder (in-batch negatives) on
     (query, positive_doc) pairs synthesized from the same topic/entity
     templates as the eval queries (held-out queries are never used for
     training).
  4. Mine hard negatives with BM25 + the trained bi-encoder and train the
     cross-encoder as a binary relevance classifier.
  5. Evaluate BM25-only / dense-only / hybrid / hybrid+reranked P@5,
     Recall@5, and MRR on held-out queries, and save results + checkpoints.

Usage:
    python -m src.train_retrieval_stack --config configs/retrieval_config.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.data.corpus import Document, generate_corpus
from src.data.queries import QueryExample, generate_eval_queries
from src.data.tokenizer import SimpleTokenizer
from src.eval.retrieval_eval import evaluate_bm25, evaluate_pipeline
from src.retrieval.bi_encoder import BiEncoder, in_batch_contrastive_loss
from src.retrieval.bm25 import BM25
from src.retrieval.cross_encoder import CrossEncoder
from src.retrieval.pipeline import RAGPipeline
from src.utils import save_json, set_seed, timer


def make_training_pairs(documents: list[Document], queries: list[QueryExample]) -> list[tuple[str, str]]:
    """(query_text, positive_doc_text) pairs for bi-encoder contrastive training."""
    doc_by_id = {d.doc_id: d for d in documents}
    pairs = []
    for q in queries:
        for did in q.relevant_doc_ids[:2]:  # a couple positives per query gives more signal without imbalancing batches too much
            pairs.append((q.text, doc_by_id[did].text))
    return pairs


def train_bi_encoder(model, tokenizer, pairs, device, epochs, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(pairs) - batch_size + 1, batch_size):
            batch = pairs[i : i + batch_size]
            q_ids = torch.tensor([tokenizer.encode(q) for q, _ in batch], dtype=torch.long, device=device)
            d_ids = torch.tensor([tokenizer.encode(d) for _, d in batch], dtype=torch.long, device=device)
            optimizer.zero_grad()
            q_emb, d_emb = model(q_ids, d_ids)
            loss = in_batch_contrastive_loss(q_emb, d_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"[bi-encoder] epoch {epoch:02d} | loss {avg:.4f}")
        history.append({"epoch": epoch, "loss": avg})
    return history


def mine_hard_negatives(pipeline, queries: list[QueryExample], documents: list[Document], num_negatives: int = 4, pool: int = 30):
    """For each query, take top pipeline (hybrid) hits that are NOT
    ground-truth relevant as hard negatives. Crucially, negatives are
    mined from the *same* retrieval distribution the cross-encoder will
    rerank at inference time (hybrid dense+BM25 candidates) rather than
    from raw BM25 alone -- training on a different negative distribution
    than the one seen at inference is a classic train/serve skew that
    causes a reranker to overfit to its training negatives and fail to
    discriminate among real candidate pools (verified empirically during
    development of this project; see README "A reranking pitfall")."""
    doc_by_id = {d.doc_id: d for d in documents}
    examples = []  # (query_text, doc_text, label)
    for q in queries:
        relevant = set(q.relevant_doc_ids)
        for did in q.relevant_doc_ids[:2]:
            examples.append((q.text, doc_by_id[did].text, 1.0))
        hits = [r.doc_id for r in pipeline.retrieve_hybrid(q.text, top_k=pool)]
        negs = [did for did in hits if did not in relevant][:num_negatives]
        for did in negs:
            examples.append((q.text, doc_by_id[did].text, 0.0))
    return examples


def train_cross_encoder(model, tokenizer, examples, device, epochs, batch_size, lr, weight_decay: float = 0.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(examples) - batch_size + 1, batch_size):
            batch = examples[i : i + batch_size]
            pair_ids = torch.tensor(
                [tokenizer.encode_pair(q, d) for q, d, _ in batch], dtype=torch.long, device=device
            )
            labels = torch.tensor([lbl for _, _, lbl in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(pair_ids)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"[cross-encoder] epoch {epoch:02d} | loss {avg:.4f}")
        history.append({"epoch": epoch, "loss": avg})
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/retrieval_config.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cpu")  # small model; CPU is fine and keeps the repo hardware-agnostic
    print(f"Using device: {device}")

    with timer("corpus generation"):
        documents = generate_corpus(num_docs=cfg["data"]["num_docs"], seed=cfg["seed"])
        train_queries = generate_eval_queries(documents, num_queries=cfg["data"]["num_train_queries"], seed=cfg["seed"] + 1)
        eval_queries = generate_eval_queries(documents, num_queries=cfg["data"]["num_eval_queries"], seed=cfg["seed"] + 999)
    all_queries = train_queries + eval_queries
    print(f"documents={len(documents)} train_queries={len(train_queries)} eval_queries={len(eval_queries)}")

    tokenizer = SimpleTokenizer.build(
        [d.text for d in documents] + [q.text for q in all_queries],
        vocab_size=cfg["tokenizer"]["vocab_size"], max_len=cfg["tokenizer"]["max_len"],
    )
    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    tokenizer.save(os.path.join(cfg["output"]["dir"], "tokenizer.json"))
    print(f"vocab_size={tokenizer.vocab_size}")

    bi_encoder = BiEncoder(tokenizer_vocab_size=tokenizer.vocab_size, **cfg["bi_encoder"]["model"]).to(device)
    bm25 = BM25().fit(documents)

    print("\n=== Training bi-encoder ===")
    train_pairs = make_training_pairs(documents, train_queries)
    with timer("bi-encoder training"):
        bi_history = train_bi_encoder(
            bi_encoder, tokenizer, train_pairs, device,
            epochs=cfg["bi_encoder"]["epochs"], batch_size=cfg["bi_encoder"]["batch_size"], lr=cfg["bi_encoder"]["lr"],
        )
    torch.save(bi_encoder.state_dict(), os.path.join(cfg["output"]["dir"], "bi_encoder.pt"))

    print("\n=== Building vector index (needed for hard-negative mining) ===")
    pipeline = RAGPipeline(documents, tokenizer, bi_encoder, cross_encoder=None, bm25=bm25, hybrid_alpha=cfg["hybrid_alpha"], device=device)
    with timer("index build"):
        pipeline.build_index()

    print("\n=== Training cross-encoder (hybrid-distribution hard negatives) ===")
    cross_encoder = CrossEncoder(tokenizer_vocab_size=tokenizer.vocab_size, **cfg["cross_encoder"]["model"]).to(device)
    ce_examples = mine_hard_negatives(pipeline, train_queries, documents, num_negatives=cfg["cross_encoder"]["num_hard_negatives"])
    print(f"cross-encoder training examples: {len(ce_examples)}")
    with timer("cross-encoder training"):
        ce_history = train_cross_encoder(
            cross_encoder, tokenizer, ce_examples, device,
            epochs=cfg["cross_encoder"]["epochs"], batch_size=cfg["cross_encoder"]["batch_size"], lr=cfg["cross_encoder"]["lr"],
            weight_decay=cfg["cross_encoder"].get("weight_decay", 0.0),
        )
    torch.save(cross_encoder.state_dict(), os.path.join(cfg["output"]["dir"], "cross_encoder.pt"))
    pipeline.cross_encoder = cross_encoder

    print("\n=== Evaluating ===")
    results = {}
    results["bm25_only"] = evaluate_bm25(bm25, eval_queries, k=5)
    results["dense_only"] = evaluate_pipeline(pipeline, eval_queries, k=5, use_hybrid=False, use_reranker=False)
    results["hybrid"] = evaluate_pipeline(pipeline, eval_queries, k=5, use_hybrid=True, use_reranker=False)
    results["hybrid_reranked"] = evaluate_pipeline(pipeline, eval_queries, k=5, use_hybrid=True, use_reranker=True)

    for name, metrics in results.items():
        print(f"{name:16s} | P@5={metrics['P@5']:.3f}  Recall@5={metrics['Recall@5']:.3f}  MRR={metrics['MRR']:.3f}")

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    save_json(
        {"config": cfg, "results": results, "bi_encoder_history": bi_history, "cross_encoder_history": ce_history,
         "num_documents": len(documents), "num_eval_queries": len(eval_queries)},
        os.path.join(cfg["output"]["results_dir"], "retrieval_eval.json"),
    )
    print(f"\nSaved results to {cfg['output']['results_dir']}/retrieval_eval.json")


if __name__ == "__main__":
    main()
