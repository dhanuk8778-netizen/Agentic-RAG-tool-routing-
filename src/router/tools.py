"""
Tool registry for the agentic router. Each tool declares a name, a short
description (used as the router's training/reference signal), and a
callable. Tools here are intentionally simple/deterministic (a calculator,
a doc-search wrapper, a stub SQL/code executor, etc.) so the whole agent
loop runs offline without external services -- the router and agent-loop
logic is what's under test, not any particular tool's implementation.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Callable

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator_tool(query: str) -> str:
    """Evaluates a simple arithmetic expression found in the query (safe AST eval, no `eval()`)."""
    match = re.search(r"[-+*/(). 0-9]{3,}", query)
    if not match:
        return "No arithmetic expression found."
    expr = match.group(0).strip()
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expr.strip()} = {result}"
    except Exception as e:  # noqa: BLE001
        return f"Could not evaluate expression: {e}"


def doc_search_tool(query: str, pipeline=None) -> str:
    """Wraps the RAG pipeline. Requires a RAGPipeline to be injected at call time."""
    if pipeline is None:
        return "[doc_search stub] no pipeline attached -- would retrieve top-5 relevant documents."
    results = pipeline.query(query, top_k=5)
    return "\n".join(f"- {r.doc_id}: {r.text[:80]}..." for r in results)


def sql_query_tool(query: str) -> str:
    return f"[sql_query stub] would translate to SQL and execute against the analytics warehouse: '{query[:60]}'"


def code_exec_tool(query: str) -> str:
    return f"[code_exec stub] would run the requested snippet in a sandboxed interpreter: '{query[:60]}'"


def summarizer_tool(query: str) -> str:
    return f"[summarizer stub] would condense the referenced content relevant to: '{query[:60]}'"


def web_search_tool(query: str) -> str:
    return f"[web_search stub] would issue an external web search for: '{query[:60]}'"


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]


TOOL_REGISTRY: dict[str, Tool] = {
    "calculator": Tool("calculator", "Evaluate arithmetic and numeric expressions.", calculator_tool),
    "doc_search": Tool("doc_search", "Search internal documents/knowledge base for relevant information.", doc_search_tool),
    "sql_query": Tool("sql_query", "Query structured/tabular data such as sales or usage metrics.", sql_query_tool),
    "code_exec": Tool("code_exec", "Execute or debug a code snippet.", code_exec_tool),
    "summarizer": Tool("summarizer", "Summarize a long piece of text or a document.", summarizer_tool),
    "web_search": Tool("web_search", "Search the public web for current information.", web_search_tool),
}

TOOL_NAMES = list(TOOL_REGISTRY.keys())
