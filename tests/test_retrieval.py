import torch

from src.data.corpus import generate_corpus
from src.data.queries import generate_eval_queries
from src.eval.metrics import precision_at_k, reciprocal_rank, recall_at_k
from src.retrieval.bi_encoder import BiEncoder, in_batch_contrastive_loss
from src.retrieval.bm25 import BM25
from src.retrieval.cross_encoder import CrossEncoder
from src.data.tokenizer import SimpleTokenizer


def test_corpus_generation_deterministic():
    docs1 = generate_corpus(num_docs=50, seed=1)
    docs2 = generate_corpus(num_docs=50, seed=1)
    assert [d.doc_id for d in docs1] == [d.doc_id for d in docs2]
    assert [d.text for d in docs1] == [d.text for d in docs2]


def test_eval_queries_have_relevant_docs():
    docs = generate_corpus(num_docs=100, seed=1)
    queries = generate_eval_queries(docs, num_queries=20, seed=2)
    for q in queries:
        assert len(q.relevant_doc_ids) > 0


def test_bm25_perfect_query_returns_self():
    docs = generate_corpus(num_docs=30, seed=1)
    bm25 = BM25().fit(docs)
    target_doc = docs[0]
    hits = bm25.search(target_doc.text, top_k=5)
    assert hits[0][0] == target_doc.doc_id  # exact text match should rank itself first


def test_metrics_precision_recall_mrr():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "e", "z"}
    assert precision_at_k(retrieved, relevant, 5) == 2 / 5
    assert recall_at_k(retrieved, relevant, 5) == 2 / 3
    assert reciprocal_rank(retrieved, relevant) == 1 / 2  # first hit "b" at rank 2


def test_bi_encoder_output_normalized():
    tok = SimpleTokenizer.build(["hello world", "goodbye world"], vocab_size=100, max_len=16)
    model = BiEncoder(tokenizer_vocab_size=tok.vocab_size, dim=16, num_layers=1, num_heads=2, ff_dim=32, max_len=16, proj_dim=8)
    ids = torch.tensor([tok.encode("hello world")])
    emb = model.encode(ids)
    norm = emb.norm(dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_contrastive_loss_lower_for_matched_pairs():
    torch.manual_seed(0)
    matched_q = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    matched_d = matched_q.clone()  # perfectly matched
    random_d = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss_matched = in_batch_contrastive_loss(matched_q, matched_d)
    loss_random = in_batch_contrastive_loss(matched_q, random_d)
    assert loss_matched.item() < loss_random.item()


def test_cross_encoder_output_shape():
    tok = SimpleTokenizer.build(["hello world", "goodbye world"], vocab_size=100, max_len=32)
    model = CrossEncoder(tokenizer_vocab_size=tok.vocab_size, dim=16, num_layers=1, num_heads=2, ff_dim=32, max_len=32)
    pair_ids = torch.tensor([tok.encode_pair("hello", "world")])
    logits = model(pair_ids)
    assert logits.shape == (1,)
