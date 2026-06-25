"""Retrieve node: pull similar Trios from the golden bucket as few-shot examples."""

from __future__ import annotations

from src.services.golden_bucket import GoldenBucket
from src.state import AgentState

# One shared instance: opening the embedded Qdrant folder is not free, and the
# embedded store allows a single client per process.
_bucket: GoldenBucket | None = None


def _get_bucket() -> GoldenBucket:
    global _bucket
    if _bucket is None:
        _bucket = GoldenBucket()
    return _bucket


def retrieve_trios(state: AgentState) -> AgentState:
    """Find the Trios most similar to the user's question."""
    obs = state["observer"]
    with obs.node("retrieve_trios"):
        trios = _get_bucket().retrieve(state["question"], k=3)
        obs.event("retrieve_trios", retrieved=len(trios), questions=[t.question for t in trios])
        return {"trios": trios}
