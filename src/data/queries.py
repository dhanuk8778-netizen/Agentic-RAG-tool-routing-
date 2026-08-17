"""
Synthetic evaluation queries with exact relevance judgments, derived from the
same topic/entity templates used in corpus.py. A query is "about" a
(topic, entity) pair; a document is judged relevant iff it shares the topic
and mentions that entity string -- giving free, exact ground truth for
computing retrieval metrics like Precision@5 without any manual labeling.

Queries paraphrase the entity name via WORD_SYNONYMS instead of quoting it
verbatim. This matters: if queries simply repeated the document's exact
wording, BM25's exact-term IDF weighting would trivially achieve near-
perfect retrieval and there would be no headroom left to demonstrate what
dense retrieval or cross-encoder reranking actually contribute. Real users
don't phrase questions with a document's exact internal terminology either
-- the lexical gap here is a deliberately-introduced, realistic difficulty,
not an accident.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.data.corpus import _ENTITIES, TOPICS, Document

_QUESTION_TEMPLATES = [
    "How do we monitor the {entity}?",
    "What is the recommended process for handling {entity} issues?",
    "Who owns the {entity} after the recent reorg?",
    "What changed with the {entity} last quarter?",
    "Is there a checklist for reviewing the {entity}?",
    "Why did the {entity} cause problems recently?",
    "What's the best practice for the {entity}?",
    "Where can I find documentation about the {entity}?",
]

# Word-level paraphrase table: maps a distinctive noun/term that appears in
# the canonical entity phrases (see corpus._ENTITIES) to 1-2 synonyms used
# when phrasing queries. Words not in this table are left as-is, so
# paraphrasing is partial (as it would be for a real user query) rather
# than a full adversarial rewrite.
WORD_SYNONYMS: dict[str, list[str]] = {
    "instance": ["compute node", "VM"], "bucket": ["storage container"], "balancer": ["traffic distributor"],
    "auto-scaling": ["elastic scaling"], "vpc": ["virtual network"], "subnet": ["network segment"],
    "kubernetes": ["container orchestration"], "cluster": ["node group"], "index": ["lookup structure"],
    "planner": ["optimizer"], "replication": ["data mirroring"], "lag": ["delay"], "pool": ["reservoir"],
    "write-ahead": ["durability"], "migration": ["schema change"], "gradient": ["optimization"],
    "descent": ["update step"], "validation": ["holdout"], "loss": ["error metric"], "pipeline": ["workflow"],
    "checkpoint": ["saved snapshot"], "hyperparameter": ["tuning setting"], "sweep": ["search run"],
    "confusion": ["error breakdown"], "matrix": ["table"], "firewall": ["access barrier"], "rule": ["policy line"],
    "token": ["credential"], "penetration": ["security"], "cve": ["known vulnerability"], "patch": ["fix"],
    "intrusion": ["unauthorized access"], "tls": ["encrypted connection"], "certificate": ["cert"],
    "bgp": ["routing protocol"], "route": ["network path"], "dns": ["domain lookup"], "packet": ["network"],
    "latency": ["response delay"], "vpn": ["secure tunnel"], "tunnel": ["encrypted link"],
    "component": ["module"], "render": ["rendering"], "cycle": ["refresh"], "bundle": ["package"],
    "css": ["styling"], "specificity": ["priority rule"], "accessibility": ["a11y"],
    "state": ["app data"], "management": ["handling"], "push": ["mobile"], "notification": ["alert"],
    "app": ["mobile app"], "background": ["async"], "task": ["job"], "battery": ["power"], "usage": ["consumption"],
    "crash": ["failure"], "report": ["log"], "deep": ["in-app"], "link": ["navigation link"],
    "build": ["compile"], "deployment": ["release"], "rollback": ["revert"], "canary": ["staged"],
    "release": ["rollout"], "artifact": ["build output"], "registry": ["storage"], "coverage": ["test"],
    "gate": ["quality check"], "infra-as-code": ["provisioning config"], "consent": ["opt-in"],
    "retention": ["storage duration"], "pii": ["personal data"], "redaction": ["masking"],
    "gdpr": ["privacy regulation"], "request": ["ask"], "encryption": ["data protection"], "key": ["secret"],
    "rate": ["throttling"], "limit": ["cap"], "pagination": ["paging"], "cursor": ["page marker"],
    "idempotency": ["duplicate-safe"], "webhook": ["callback"], "payload": ["message body"],
    "versioning": ["revisioning"], "scheme": ["numbering"], "error": ["failure"], "envelope": ["wrapper format"],
    "leader": ["primary node"], "election": ["selection"], "consensus": ["agreement"], "protocol": ["algorithm"],
    "partition": ["network split"], "tolerance": ["resilience"], "message": ["event"], "queue": ["broker"],
    "eventual": ["delayed"], "consistency": ["sync guarantee"], "sharding": ["partitioning"],
    "roadmap": ["plan"], "milestone": ["target date"], "story": ["ticket"], "flag": ["toggle"],
    "adoption": ["usage rate"], "metric": ["KPI"], "stakeholder": ["exec"], "review": ["check-in"],
    "launch": ["rollout"], "checklist": ["run-through"], "onboarding": ["ramp-up"], "pto": ["time off"],
    "performance": ["review cycle"], "benefits": ["perks"], "enrollment": ["sign-up"], "remote": ["distributed"],
    "work": ["work arrangement"], "compliance": ["policy"], "training": ["certification"],
    "quarterly": ["end-of-quarter"], "close": ["closing process"], "expense": ["spend"], "revenue": ["income"],
    "recognition": ["booking"], "budget": ["spend plan"], "variance": ["deviation"], "audit": ["review"],
    "trail": ["history log"], "cost": ["spend"], "center": ["cost bucket"], "contract": ["agreement"],
    "clause": ["provision"], "regulatory": ["compliance"], "filing": ["submission"], "nda": ["confidentiality agreement"],
    "policy": ["rule set"], "exception": ["waiver"], "vendor": ["supplier"], "agreement": ["contract"],
    "ticket": ["support case"], "escalation": ["priority bump"], "sla": ["service commitment"], "breach": ["violation"],
    "resolution": ["fix time"], "customer": ["client"], "satisfaction": ["CSAT"], "score": ["rating"],
    "knowledge": ["help"], "article": ["doc"], "refund": ["reimbursement"], "lead": ["prospect"],
    "qualification": ["vetting"], "quota": ["target"], "attainment": ["achievement"], "discount": ["price break"],
    "approval": ["sign-off"], "renewal": ["contract extension"], "forecast": ["projection"], "deal": ["opportunity"],
    "desk": ["approval team"], "conversion": ["signup rate"], "funnel": ["pipeline"], "attribution": ["credit"],
    "model": ["framework"], "campaign": ["ad campaign"], "spend": ["budget"], "churn": ["cancellation"],
    "cohort": ["user group"], "engagement": ["activity"], "inventory": ["stock"], "turnover": ["restock rate"],
    "shipment": ["delivery"], "tracking": ["monitoring"], "warehouse": ["storage facility"], "demand": ["order volume"],
    "procurement": ["purchasing"], "rack": ["server rack"], "temperature": ["thermal reading"], "firmware": ["device software"],
    "power": ["electricity"], "draw": ["consumption"], "disk": ["storage drive"], "failure": ["outage"],
    "cooling": ["HVAC"], "asset": ["equipment"], "s3": ["object storage"], "ec2": ["cloud compute"],
    "connection": ["session"], "primary": ["main"], "schema": ["data model"], "user": ["end user"],
    "a/b": ["split"], "test": ["experiment"], "feature": ["capability"],
}


def _paraphrase(entity: str, rng: random.Random, prob: float = 0.75) -> str:
    words = entity.split()
    out = []
    for w in words:
        syns = WORD_SYNONYMS.get(w.lower())
        if syns and rng.random() < prob:
            out.append(rng.choice(syns))
        else:
            out.append(w)
    return " ".join(out)


@dataclass
class QueryExample:
    query_id: str
    text: str
    topic: str
    entity: str
    relevant_doc_ids: list[str]


def generate_eval_queries(
    documents: list[Document], num_queries: int = 300, seed: int = 123
) -> list[QueryExample]:
    rng = random.Random(seed)

    # Index docs by (topic, entity mention) for exact relevance lookup.
    by_topic: dict[str, list[Document]] = {}
    for d in documents:
        by_topic.setdefault(d.topic, []).append(d)

    queries: list[QueryExample] = []
    for i in range(num_queries):
        topic = TOPICS[i % len(TOPICS)]
        entity = rng.choice(_ENTITIES[topic])
        tmpl = rng.choice(_QUESTION_TEMPLATES)
        query_phrase = _paraphrase(entity, rng)
        text = tmpl.format(entity=query_phrase)

        relevant = [d.doc_id for d in by_topic.get(topic, []) if entity in d.text]
        if len(relevant) < 1:
            # guarantee at least one relevant doc by relaxing to topic-only match
            relevant = [d.doc_id for d in by_topic.get(topic, [])][:3]

        queries.append(QueryExample(
            query_id=f"q_{i:04d}", text=text, topic=topic, entity=entity,
            relevant_doc_ids=relevant,
        ))
    return queries
