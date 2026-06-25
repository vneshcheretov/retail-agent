"""Retrieval eval for the golden bucket: measures recall@1, recall@3 and MRR.

Each test case is a paraphrased question (worded differently from the stored
Trio) plus the exact stored question we expect to retrieve. We check whether the
target shows up in the top-k results.

Uses the real GoldenBucket when the qdrant folder is free; otherwise falls back
to an in-memory mirror seeded from seed_trios.json (same embedder + COSINE, so
results match the service).

    python scripts/eval_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings import settings  # noqa: E402

# (paraphrased query, exact stored target question)
TEST_CASES: list[tuple[str, str]] = [
    ("who are our biggest spenders?", "Who are our top 5 customers by total spend?"),
    ("which categories make us the most money?", "Which product categories bring the most revenue?"),
    ("what's the typical basket size?", "What is the average order value?"),
    ("how many shoppers do we have in each nation?", "How many customers do we have per country? Top 10."),
    ("which product categories see the most returns relative to sales?", "Which product categories have the highest return rate?"),
    ("which single products get sent back most often?", "Which products are returned the most?"),
    ("total sales since the very beginning", "What is our total revenue all time?"),
    ("how old are our buyers on average?", "What is the average age of our customers?"),
    ("how many customers does each acquisition channel bring?", "How many customers come from each traffic source?"),
    ("best sellers by quantity sold", "What are the 10 best selling products by units?"),
    ("which brands earn the most revenue?", "Which brands generate the most revenue?"),
    ("monthly sales for this year", "What is our monthly revenue this year?"),
    ("share of customers who buy again", "What share of customers are repeat buyers?"),
    ("how long until orders arrive on average?", "What is the average time from order to delivery in days?"),
    ("revenue contribution by department", "How much revenue does each department make?"),
    ("which countries bring the most sales?", "Which countries generate the most revenue?"),
    ("which US states have the most shoppers?", "What are the top 10 US states by number of customers?"),
    ("average money spent per customer", "How much does the average customer spend?"),
]

K = 3


def build_retriever():
    """Return (retrieve_fn, label). retrieve_fn(query, k) -> list[question]."""
    try:
        from src.services.golden_bucket import GoldenBucket

        bucket = GoldenBucket()
        return (lambda q, k: [t.question for t in bucket.retrieve(q, k)], "GoldenBucket (qdrant_data)")
    except Exception:
        from qdrant_client import QdrantClient, models

        from src.services.embeddings import get_embedder

        embedder = get_embedder()
        trios = json.loads(settings.seed_trios_path.read_text(encoding="utf-8"))
        client = QdrantClient(":memory:")
        client.create_collection(
            "eval",
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        client.upsert("eval", points=[
            models.PointStruct(id=i, vector=embedder.embed([t["question"]])[0], payload=t)
            for i, t in enumerate(trios)
        ])

        def retrieve(q: str, k: int) -> list[str]:
            hits = client.query_points("eval", query=embedder.embed([q])[0], limit=k).points
            return [h.payload["question"] for h in hits]

        return (retrieve, "in-memory mirror (qdrant locked)")


def main() -> None:
    corpus = {t["question"] for t in json.loads(settings.seed_trios_path.read_text(encoding="utf-8"))}
    missing = [t for _, t in TEST_CASES if t not in corpus]
    if missing:
        raise SystemExit(f"Target(s) not in corpus: {missing}")

    retrieve, label = build_retriever()
    print(f"Retriever: {label}\nCases: {len(TEST_CASES)}, k={K}\n")

    hits_at_1 = hits_at_k = 0
    reciprocal_ranks = 0.0
    for query, target in TEST_CASES:
        results = retrieve(query, K)
        rank = results.index(target) + 1 if target in results else 0
        hits_at_1 += rank == 1
        hits_at_k += rank != 0
        reciprocal_ranks += 1 / rank if rank else 0
        mark = f"rank {rank}" if rank else "MISS"
        print(f"  [{'OK ' if rank else 'XX '}] {mark:7} {query!r}")
        if not rank:
            print(f"           expected: {target!r}")
            print(f"           got:      {results}")

    n = len(TEST_CASES)
    print(f"\nrecall@1 = {hits_at_1}/{n} = {hits_at_1 / n:.0%}")
    print(f"recall@{K} = {hits_at_k}/{n} = {hits_at_k / n:.0%}")
    print(f"MRR      = {reciprocal_ranks / n:.3f}")


if __name__ == "__main__":
    main()
