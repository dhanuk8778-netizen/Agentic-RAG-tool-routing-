"""
Synthetic multi-intent queries for training/evaluating the semantic router.
Each query is built from one or more single-tool "intent fragments" stitched
together, giving an exact multi-label ground truth (which tools the query
actually requires) for free -- the same free-ground-truth-via-generation
trick used for the retrieval corpus in src/data/queries.py.

~30% of queries combine two intent fragments (e.g. "search the docs for X,
then calculate Y") to specifically exercise multi-intent routing, matching
the reported 250-query multi-intent router evaluation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

_INTENT_FRAGMENTS: dict[str, list[str]] = {
    "calculator": [
        "what is {a} * {b}", "compute {a} + {b}", "calculate {a} percent of {b}",
        "what's the sum of {a} and {b}", "divide {a} by {b}",
    ],
    "doc_search": [
        "find the documentation on {topic}", "search our knowledge base for {topic}",
        "look up the internal wiki page about {topic}", "what do our docs say about {topic}",
    ],
    "sql_query": [
        "how many {entity} did we have last month", "what was the total {entity} this quarter",
        "pull the count of {entity} from the database", "query the {entity} table for last week's totals",
    ],
    "code_exec": [
        "run this snippet and tell me the output", "debug why this function throws an error",
        "execute the script and show the result", "test whether this code compiles",
    ],
    "summarizer": [
        "summarize the attached report", "give me a short summary of this document",
        "condense this into three bullet points", "what's the tl;dr of this thread",
    ],
    "web_search": [
        "what's the latest news on {topic}", "search the web for current {topic} pricing",
        "look up today's {topic} headlines", "find recent articles about {topic}",
    ],
}

_TOPICS = ["Kubernetes autoscaling", "GDPR compliance", "the Q3 roadmap", "vendor contracts", "the onboarding process", "API rate limits"]
_ENTITIES = ["support tickets", "new signups", "failed deployments", "refund requests", "active users"]

_CONNECTORS = [" and then ", ", then ", " -- also, ", ". Separately, "]


@dataclass
class RouterQueryExample:
    query_id: str
    text: str
    tool_labels: list[str]  # ground-truth multi-label tool set


def _fill(fragment: str, rng: random.Random) -> str:
    return fragment.format(
        a=rng.randint(2, 500), b=rng.randint(2, 500),
        topic=rng.choice(_TOPICS), entity=rng.choice(_ENTITIES),
    )


def generate_router_queries(num_queries: int = 250, multi_intent_fraction: float = 0.3, seed: int = 7) -> list[RouterQueryExample]:
    rng = random.Random(seed)
    tool_names = list(_INTENT_FRAGMENTS.keys())
    examples: list[RouterQueryExample] = []

    for i in range(num_queries):
        is_multi = rng.random() < multi_intent_fraction
        n_intents = 2 if is_multi else 1
        chosen_tools = rng.sample(tool_names, n_intents)
        parts = [_fill(rng.choice(_INTENT_FRAGMENTS[t]), rng) for t in chosen_tools]
        if n_intents == 1:
            text = parts[0].capitalize() + "?"
        else:
            connector = rng.choice(_CONNECTORS)
            text = parts[0].capitalize() + connector + parts[1] + "."
        examples.append(RouterQueryExample(query_id=f"r_{i:04d}", text=text, tool_labels=sorted(chosen_tools)))
    return examples


# --- Shifted-domain queries: terse, Slack-shorthand-style phrasing of the
# same underlying intents, used to demonstrate QLoRA/PEFT domain adaptation
# (src/peft/finetune_router.py). A router trained only on the fuller
# question-style phrasing above sees a real (if synthetic) distribution
# shift here -- abbreviations, dropped function words, no punctuation.
_SHIFTED_FRAGMENTS: dict[str, list[str]] = {
    "calculator": ["calc {a}*{b}", "{a}+{b}=?", "qk math {a} / {b}"],
    "doc_search": ["docs on {topic}?", "kb search {topic}", "wiki: {topic}"],
    "sql_query": ["count {entity} last mo", "db: {entity} totals", "how many {entity} qtd"],
    "code_exec": ["run this snippet", "exec + debug pls", "test this code"],
    "summarizer": ["tldr this doc", "summarize pls", "3 bullets from this"],
    "web_search": ["latest {topic} news?", "google {topic} pricing", "web: {topic} today"],
}


def generate_shifted_domain_queries(num_queries: int = 150, multi_intent_fraction: float = 0.2, seed: int = 99) -> list[RouterQueryExample]:
    rng = random.Random(seed)
    tool_names = list(_SHIFTED_FRAGMENTS.keys())
    examples: list[RouterQueryExample] = []
    for i in range(num_queries):
        is_multi = rng.random() < multi_intent_fraction
        n_intents = 2 if is_multi else 1
        chosen_tools = rng.sample(tool_names, n_intents)
        parts = [_fill(rng.choice(_SHIFTED_FRAGMENTS[t]), rng) for t in chosen_tools]
        text = " / ".join(parts)
        examples.append(RouterQueryExample(query_id=f"s_{i:04d}", text=text, tool_labels=sorted(chosen_tools)))
    return examples
