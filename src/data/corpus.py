"""
Synthetic multi-domain document corpus.

Real production RAG corpora (internal wikis, support tickets, contracts)
can't be vendored in a public repo. To keep this project fully offline and
reproducible -- and, crucially, to have *known ground-truth relevance* so
retrieval quality (P@5) can actually be measured rather than eyeballed --
this module procedurally generates a corpus of short documents across a
fixed set of topics, each built from a topic-specific template bank plus
randomized entities/facts. Because generation is template-grounded, we know
exactly which documents are relevant to a query synthesized from the same
topic, giving exact relevance judgments for free.

Swap in `load_real_corpus()` (stub below) to point the same downstream
pipeline at a real document set -- nothing else in src/retrieval changes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

TOPICS = [
    "cloud_infrastructure", "database_systems", "machine_learning",
    "cybersecurity", "networking", "frontend_engineering",
    "mobile_development", "devops_cicd", "data_privacy", "api_design",
    "distributed_systems", "product_management", "hr_policy",
    "finance_reporting", "legal_compliance", "customer_support",
    "sales_process", "marketing_analytics", "supply_chain", "hardware_ops",
]

# Per-topic vocabulary used to fill templates -- gives each topic a
# distinct, consistent "signature" that a dense retriever can learn to
# separate, similar in spirit to how real domain corpora cluster.
_ENTITIES = {
    "cloud_infrastructure": ["EC2 instance", "S3 bucket", "load balancer", "auto-scaling group", "VPC subnet", "Kubernetes cluster"],
    "database_systems": ["primary index", "query planner", "replication lag", "connection pool", "write-ahead log", "schema migration"],
    "machine_learning": ["gradient descent", "validation loss", "feature pipeline", "model checkpoint", "hyperparameter sweep", "confusion matrix"],
    "cybersecurity": ["firewall rule", "access token", "penetration test", "CVE patch", "intrusion alert", "TLS certificate"],
    "networking": ["BGP route", "DNS record", "packet loss", "latency spike", "subnet mask", "VPN tunnel"],
    "frontend_engineering": ["component tree", "render cycle", "bundle size", "CSS specificity", "accessibility audit", "state management"],
    "mobile_development": ["push notification", "app store review", "background task", "battery usage", "crash report", "deep link"],
    "devops_cicd": ["build pipeline", "deployment rollback", "canary release", "artifact registry", "test coverage gate", "infra-as-code"],
    "data_privacy": ["consent record", "data retention policy", "PII redaction", "GDPR request", "audit log", "encryption key"],
    "api_design": ["rate limit", "pagination cursor", "idempotency key", "webhook payload", "versioning scheme", "error envelope"],
    "distributed_systems": ["leader election", "consensus protocol", "partition tolerance", "message queue", "eventual consistency", "sharding key"],
    "product_management": ["roadmap milestone", "user story", "feature flag", "adoption metric", "stakeholder review", "launch checklist"],
    "hr_policy": ["onboarding checklist", "PTO request", "performance review", "benefits enrollment", "remote work policy", "compliance training"],
    "finance_reporting": ["quarterly close", "expense report", "revenue recognition", "budget variance", "audit trail", "cost center"],
    "legal_compliance": ["contract clause", "regulatory filing", "NDA template", "compliance audit", "policy exception", "vendor agreement"],
    "customer_support": ["ticket escalation", "SLA breach", "resolution time", "customer satisfaction score", "knowledge base article", "refund request"],
    "sales_process": ["pipeline stage", "lead qualification", "quota attainment", "discount approval", "renewal forecast", "deal desk"],
    "marketing_analytics": ["conversion funnel", "attribution model", "campaign spend", "A/B test", "churn cohort", "engagement metric"],
    "supply_chain": ["inventory turnover", "shipment tracking", "vendor lead time", "warehouse capacity", "demand forecast", "procurement order"],
    "hardware_ops": ["rack temperature", "firmware update", "power draw", "disk failure rate", "cooling system", "asset inventory"],
}

_TEMPLATES = [
    "The {entity} was updated after the team identified an issue during the {period} review.",
    "Engineers documented the {entity} configuration to help new hires ramp up faster.",
    "A recent incident traced back to an unexpected change in the {entity}, prompting a policy update.",
    "This guide explains how to monitor the {entity} and respond to alerts within {period}.",
    "The {period} report highlighted a steady improvement in {entity} after the migration.",
    "Best practices for managing {entity} include automated checks and a documented rollback plan.",
    "The team compared two approaches to the {entity} problem and chose the one with lower long-term cost.",
    "During the {period} planning cycle, ownership of the {entity} moved to a dedicated working group.",
    "A checklist was created to standardize how the {entity} is reviewed before every release.",
    "Historical data on the {entity} shows a clear seasonal pattern worth accounting for in forecasts.",
]

_PERIODS = ["Q1", "Q2", "Q3", "Q4", "weekly", "monthly", "post-incident", "annual"]


@dataclass
class Document:
    doc_id: str
    topic: str
    title: str
    text: str


def _make_title(topic: str, entity: str) -> str:
    return f"{topic.replace('_', ' ').title()}: notes on {entity}"


def generate_corpus(num_docs: int = 5200, seed: int = 42) -> list[Document]:
    """Deterministically generate `num_docs` synthetic documents spread
    roughly evenly across TOPICS, each with 2-4 template sentences."""
    rng = random.Random(seed)
    docs: list[Document] = []
    for i in range(num_docs):
        topic = TOPICS[i % len(TOPICS)]
        entities = _ENTITIES[topic]
        n_sent = rng.randint(2, 4)
        sentences = []
        chosen_entity = rng.choice(entities)
        for _ in range(n_sent):
            tmpl = rng.choice(_TEMPLATES)
            entity = rng.choice(entities)
            period = rng.choice(_PERIODS)
            sentences.append(tmpl.format(entity=entity, period=period))
        text = " ".join(sentences)
        title = _make_title(topic, chosen_entity)
        docs.append(Document(doc_id=f"doc_{i:05d}", topic=topic, title=title, text=f"{title}. {text}"))
    rng.shuffle(docs)
    return docs


def load_real_corpus(path: str) -> list[Document]:
    """Stub for wiring in a real corpus: expects a JSONL file with one
    {"doc_id": ..., "topic": ..., "title": ..., "text": ...} object per
    line (topic/title are optional metadata, text is required)."""
    import json
    docs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(
                doc_id=obj["doc_id"], topic=obj.get("topic", ""),
                title=obj.get("title", ""), text=obj["text"],
            ))
    return docs
