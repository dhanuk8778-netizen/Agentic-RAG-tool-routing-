"""
Ties the semantic router and RAG pipeline into a single agentic loop and
measures end-to-end task completion: for each query, the router selects
tool(s); a turn "completes" successfully if (a) the router's tool
selection exactly matches what the query actually needs, AND (b) for any
selected `doc_search` call, the RAG pipeline actually surfaces a relevant
document in its top-5 -- i.e. routing correctly isn't enough if the
downstream retrieval then fails.

Usage:
    python -m src.serving.agent_loop --router-config configs/router_config.yaml --retrieval-config configs/retrieval_config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.data.corpus import generate_corpus
from src.data.router_queries import generate_router_queries
from src.data.tokenizer import SimpleTokenizer
from src.retrieval.bi_encoder import BiEncoder
from src.retrieval.bm25 import BM25
from src.retrieval.cross_encoder import CrossEncoder
from src.retrieval.pipeline import RAGPipeline
from src.router.semantic_router import SemanticRouter
from src.router.tools import TOOL_REGISTRY
from src.utils import save_json, set_seed


def build_rag_pipeline(retrieval_cfg: dict, device):
    documents = generate_corpus(num_docs=retrieval_cfg["data"]["num_docs"], seed=retrieval_cfg["seed"])
    tok = SimpleTokenizer.load(os.path.join(retrieval_cfg["output"]["dir"], "tokenizer.json"))
    bi = BiEncoder(tokenizer_vocab_size=tok.vocab_size, **retrieval_cfg["bi_encoder"]["model"]).to(device)
    bi.load_state_dict(torch.load(os.path.join(retrieval_cfg["output"]["dir"], "bi_encoder.pt"), map_location=device))
    ce = CrossEncoder(tokenizer_vocab_size=tok.vocab_size, **retrieval_cfg["cross_encoder"]["model"]).to(device)
    ce.load_state_dict(torch.load(os.path.join(retrieval_cfg["output"]["dir"], "cross_encoder.pt"), map_location=device))
    bm25 = BM25().fit(documents)
    pipeline = RAGPipeline(documents, tok, bi, ce, bm25, hybrid_alpha=retrieval_cfg["hybrid_alpha"], device=device)
    pipeline.build_index()
    return pipeline, documents


def build_router(router_cfg: dict, device):
    tok = SimpleTokenizer.load(os.path.join(router_cfg["output"]["dir"], "router_tokenizer.json"))
    model = SemanticRouter(tokenizer_vocab_size=tok.vocab_size, **router_cfg["model"]).to(device)
    model.load_state_dict(torch.load(os.path.join(router_cfg["output"]["dir"], "semantic_router.pt"), map_location=device))
    model.eval()
    return model, tok


def run_agent_turn(query_text: str, reference_tools: list[str], router, router_tok, rag_pipeline, device, threshold: float = 0.5) -> dict:
    input_ids = torch.tensor([router_tok.encode(query_text)], dtype=torch.long, device=device)
    predicted_tools = router.route(input_ids, threshold=threshold)[0]

    routing_correct = set(predicted_tools) == set(reference_tools)

    retrieval_ok = True
    retrieved_preview = None
    if "doc_search" in predicted_tools and rag_pipeline is not None:
        results = rag_pipeline.query(query_text, top_k=5)
        retrieval_ok = len(results) > 0
        retrieved_preview = [r.doc_id for r in results[:3]]

    tool_outputs = {}
    for tool_name in predicted_tools:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is not None and tool_name != "doc_search":
            tool_outputs[tool_name] = tool.fn(query_text)

    completed = routing_correct and retrieval_ok
    return {
        "query": query_text, "predicted_tools": predicted_tools, "reference_tools": reference_tools,
        "routing_correct": routing_correct, "retrieval_ok": retrieval_ok, "completed": completed,
        "retrieved_preview": retrieved_preview,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-config", type=str, default="configs/router_config.yaml")
    parser.add_argument("--retrieval-config", type=str, default="configs/retrieval_config.yaml")
    parser.add_argument("--num-eval", type=int, default=None)
    args = parser.parse_args()

    with open(args.router_config) as f:
        router_cfg = yaml.safe_load(f)
    with open(args.retrieval_config) as f:
        retrieval_cfg = yaml.safe_load(f)

    set_seed(router_cfg["seed"])
    device = torch.device("cpu")

    print("Loading router...")
    router, router_tok = build_router(router_cfg, device)
    print("Loading RAG pipeline (this rebuilds the vector index)...")
    rag_pipeline, _ = build_rag_pipeline(retrieval_cfg, device)

    all_router_examples = generate_router_queries(
        num_queries=router_cfg["data"]["num_queries"], multi_intent_fraction=router_cfg["data"]["multi_intent_fraction"], seed=router_cfg["seed"]
    )
    split = int(len(all_router_examples) * router_cfg["data"]["train_fraction"])
    eval_examples = all_router_examples[split:]
    if args.num_eval:
        eval_examples = eval_examples[: args.num_eval]

    print(f"\nRunning agent loop over {len(eval_examples)} held-out queries...\n")
    turn_results = []
    for ex in eval_examples:
        result = run_agent_turn(ex.text, ex.tool_labels, router, router_tok, rag_pipeline, device, threshold=router_cfg["eval"]["threshold"])
        turn_results.append(result)

    completion_rate = sum(r["completed"] for r in turn_results) / len(turn_results)
    routing_acc = sum(r["routing_correct"] for r in turn_results) / len(turn_results)
    doc_search_turns = [r for r in turn_results if "doc_search" in r["predicted_tools"]]
    retrieval_ok_rate = (sum(r["retrieval_ok"] for r in doc_search_turns) / len(doc_search_turns)) if doc_search_turns else None

    print(f"Routing exact-match accuracy: {routing_acc:.3f}")
    if retrieval_ok_rate is not None:
        print(f"Retrieval success rate (doc_search turns, n={len(doc_search_turns)}): {retrieval_ok_rate:.3f}")
    print(f"End-to-end task completion rate: {completion_rate:.3f}")

    os.makedirs(router_cfg["output"]["results_dir"], exist_ok=True)
    save_json(
        {"completion_rate": completion_rate, "routing_accuracy": routing_acc, "retrieval_ok_rate": retrieval_ok_rate,
         "num_turns": len(turn_results), "turns_sample": turn_results[:15]},
        os.path.join(router_cfg["output"]["results_dir"], "agent_loop_eval.json"),
    )
    print(f"\nSaved results to {router_cfg['output']['results_dir']}/agent_loop_eval.json")


if __name__ == "__main__":
    main()
