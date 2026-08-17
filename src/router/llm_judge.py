"""
LLM-as-a-judge for tool-selection validation.

Two backends behind one interface:
  - `AnthropicJudge`: calls the real Claude API (api.anthropic.com) to
    grade whether the router's predicted tool(s) are a reasonable choice
    for the query, given each tool's description -- a genuine LLM-as-judge
    setup, used when `ANTHROPIC_API_KEY` is set in the environment.
  - `RuleBasedJudge`: a deterministic fallback (exact predicted/reference
    tool-set match) so the router evaluation pipeline, CI, and this repo's
    demo results are runnable and reproducible with zero API dependency.

`get_judge()` picks the API judge automatically when a key is available,
otherwise falls back with a printed notice -- nothing else in the router
eval code needs to know which backend is active.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.router.tools import TOOL_REGISTRY

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
JUDGE_MODEL = "claude-sonnet-5"


@dataclass
class JudgeVerdict:
    correct: bool
    rationale: str


class RuleBasedJudge:
    """Deterministic fallback: predicted tool set must exactly match the
    reference (ground-truth) tool set. Strict, reproducible, offline --
    used for CI and for this repo's committed demo results."""

    name = "rule_based"

    def judge(self, query: str, predicted_tools: list[str], reference_tools: list[str]) -> JudgeVerdict:
        correct = set(predicted_tools) == set(reference_tools)
        rationale = "exact tool-set match" if correct else f"predicted {predicted_tools} != reference {reference_tools}"
        return JudgeVerdict(correct=correct, rationale=rationale)


class AnthropicJudge:
    """Real LLM-as-a-judge via the Claude API. Given the query, the tools
    the router selected, and the descriptions of all available tools (NOT
    the reference labels -- the judge grades reasonableness independently,
    the way a real deployment would since ground truth isn't available at
    serving time), asks the model for a structured correct/incorrect
    verdict plus a one-line rationale.
    """

    name = "anthropic_llm_judge"

    def __init__(self, api_key: str | None = None, model: str = JUDGE_MODEL, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.model = model
        self.timeout = timeout

    def _tool_menu(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in TOOL_REGISTRY.values())

    def judge(self, query: str, predicted_tools: list[str], reference_tools: list[str] | None = None) -> JudgeVerdict:
        prompt = (
            "You are grading a tool-routing system for an AI agent. Given the user "
            "query and the available tools, decide whether the SELECTED tool(s) are "
            "a reasonable, sufficient choice to answer the query. Respond with strict "
            f'JSON only: {{"correct": true|false, "rationale": "<one short sentence>"}}.\n\n'
            f"Available tools:\n{self._tool_menu()}\n\n"
            f"User query: {query}\n"
            f"Selected tool(s): {', '.join(predicted_tools) if predicted_tools else '(none)'}\n"
        )
        body = json.dumps({
            "model": self.model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_API_URL, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
            text = "".join(b["text"] for b in payload.get("content", []) if b.get("type") == "text")
            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return JudgeVerdict(correct=bool(parsed["correct"]), rationale=parsed.get("rationale", ""))
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as e:
            # Network unavailable, bad response shape, etc. -- fail closed to the
            # deterministic judge rather than silently miscounting an eval run.
            return RuleBasedJudge().judge(query, predicted_tools, reference_tools or [])


def get_judge():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return AnthropicJudge(api_key=api_key)
        except RuntimeError:
            pass
    print("[llm_judge] ANTHROPIC_API_KEY not set -- using RuleBasedJudge fallback "
          "(exact tool-set match). Set the env var to use real LLM-as-a-judge grading.")
    return RuleBasedJudge()
