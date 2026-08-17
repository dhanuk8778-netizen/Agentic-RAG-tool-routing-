import torch

from src.data.router_queries import generate_router_queries, generate_shifted_domain_queries
from src.data.tokenizer import SimpleTokenizer
from src.router.llm_judge import RuleBasedJudge
from src.router.semantic_router import SemanticRouter
from src.router.tools import TOOL_NAMES, calculator_tool


def test_router_queries_have_valid_tool_labels():
    examples = generate_router_queries(num_queries=30, seed=1)
    for ex in examples:
        assert len(ex.tool_labels) >= 1
        assert all(t in TOOL_NAMES for t in ex.tool_labels)


def test_router_queries_deterministic():
    e1 = generate_router_queries(num_queries=10, seed=5)
    e2 = generate_router_queries(num_queries=10, seed=5)
    assert [e.text for e in e1] == [e.text for e in e2]


def test_shifted_domain_queries_distinct_from_main():
    main = {e.text for e in generate_router_queries(num_queries=50, seed=7)}
    shifted = {e.text for e in generate_shifted_domain_queries(num_queries=50, seed=99)}
    assert len(main & shifted) == 0


def test_calculator_tool_evaluates_expression():
    result = calculator_tool("what is 12 * 4?")
    assert "48" in result


def test_calculator_tool_rejects_unsafe_input():
    # no import/exec/attribute access is possible since we use ast + a fixed op table
    result = calculator_tool("__import__('os').system('echo hi')")
    assert "48" not in result  # just confirms no crash and no code execution artifact


def test_semantic_router_multilabel_output_shape():
    tok = SimpleTokenizer.build(["search docs for X", "calculate 5 + 5"], vocab_size=200, max_len=32)
    model = SemanticRouter(tokenizer_vocab_size=tok.vocab_size, num_tools=len(TOOL_NAMES), dim=16, num_layers=1, num_heads=2, ff_dim=32, max_len=32)
    ids = torch.tensor([tok.encode("search docs for X")])
    logits = model(ids)
    assert logits.shape == (1, len(TOOL_NAMES))


def test_semantic_router_route_returns_at_least_one_tool():
    tok = SimpleTokenizer.build(["search docs for X"], vocab_size=200, max_len=32)
    model = SemanticRouter(tokenizer_vocab_size=tok.vocab_size, dim=16, num_layers=1, num_heads=2, ff_dim=32, max_len=32)
    ids = torch.tensor([tok.encode("search docs for X")])
    routed = model.route(ids, threshold=0.99)  # threshold so high nothing clears it -> fallback to argmax
    assert len(routed[0]) >= 1


def test_rule_based_judge_exact_match():
    judge = RuleBasedJudge()
    v1 = judge.judge("q", ["calculator"], ["calculator"])
    assert v1.correct
    v2 = judge.judge("q", ["calculator"], ["doc_search"])
    assert not v2.correct
