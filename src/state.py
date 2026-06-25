"""Shared state passed between graph nodes.

One ``AgentState`` object flows through the pipeline. Each node reads what it
needs and writes its own fields. Objects like the observer and retrieved Trios
live here too; the graph runs in-memory per turn, so they need not be JSON.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.services.golden_bucket import Trio
from src.services.observability import RunObserver


class AgentState(TypedDict, total=False):
    """Everything the agent builds up while answering one question."""

    # Input
    question: str
    user: str
    history: list[dict[str, str]]  # prior turns, for follow-up context
    observer: RunObserver

    # guardrail
    allowed: bool
    intent: str  # analysis | smalltalk | blocked
    refusal: str

    # retrieve_trios
    trios: list[Trio]

    # generate_sql / execute_sql
    sql: str
    attempts: int
    error: str | None
    rows: list[dict[str, Any]]  # raw rows, may contain PII
    pii_columns: list[str]  # output columns the SQL author flagged as PII

    # mask_pii
    masked_rows: list[dict[str, Any]]

    # report
    answer: str
