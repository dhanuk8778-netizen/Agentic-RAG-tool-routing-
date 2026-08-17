"""
Evaluate retrieval quality (P@5, Recall@5, MRR@10) across four
configurations: BM25-only, dense-only, hybrid (dense+BM25), and
hybrid+cross-encoder-reranked -- so the value each stage adds is visible.
"""
from __future__ import annotations

from src.data.queries import QueryExample
from src.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.retrieval.bm25 import BM25
from src.retrieval.pipeline import RAGPipeline


def evaluate_bm25(bm25: BM25, queries: list[QueryExample], k: int = 5) -> dict:
    p_list, r_list, mrr_list = [], [], []
    for q in queries:
        relevant = set(q.relevant_doc_ids)
        hits = [did for did, _ in bm25.search(q.text, top_k=10)]
        p_list.append(precision_at_k(hits, relevant, k))
        r_list.append(recall_at_k(hits, relevant, k))
        mrr_list.append(reciprocal_rank(hits, relevant))
    return _summarize(p_list, r_list, mrr_list)


def evaluate_pipeline(
    pipeline: RAGPipeline, queries: list[QueryExample], k: int = 5,
    use_hybrid: bool = True, use_reranker: bool = True, candidate_pool: int = 50,
) -> dict:
    p_list, r_list, mrr_list = [], [], []
    for q in queries:
        relevant = set(q.relevant_doc_ids)
        results = pipeline.query(
            q.text, top_k=10, candidate_pool=candidate_pool,
            use_hybrid=use_hybrid, use_reranker=use_reranker,
        )
        hits = [r.doc_id for r in results]
        p_list.append(precision_at_k(hits, relevant, k))
        r_list.append(recall_at_k(hits, relevant, k))
        mrr_list.append(reciprocal_rank(hits, relevant))
    return _summarize(p_list, r_list, mrr_list)


def _summarize(p_list: list[float], r_list: list[float], mrr_list: list[float]) -> dict:
    n = max(len(p_list), 1)
    return {
        "P@5": sum(p_list) / n,
        "Recall@5": sum(r_list) / n,
        "MRR": sum(mrr_list) / n,
        "num_queries": len(p_list),
    }
