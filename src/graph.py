"""Assemble the agent pipeline as a LangGraph state machine.

Flow:
    guardrail -> (blocked? END)
              -> retrieve_trios -> generate_sql -> execute_sql
              -> (failed & retries left? back to generate_sql)
              -> mask_pii -> report -> END

The retry edge is the self-correction loop: a failed or empty query goes back to
generate_sql once, with the error attached, before the agent gives up.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.nodes.execute_sql import execute_sql
from src.nodes.generate_sql import generate_sql
from src.nodes.guardrail import guardrail
from src.nodes.mask_pii import mask_pii
from src.nodes.report import report
from src.nodes.retrieve import retrieve_trios
from src.state import AgentState

MAX_ATTEMPTS = 2


def _after_guardrail(state: AgentState) -> str:
    """Stop early if the question was blocked."""
    return "retrieve_trios" if state.get("allowed") else END


def _after_execute(state: AgentState) -> str:
    """Retry once on error/empty result, otherwise move on to masking."""
    if state.get("error") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "generate_sql"
    return "mask_pii"


def build_graph(store=None):
    """Build and compile the agent graph.

    ``store`` is an optional LangGraph store for per-user long-term memory; the
    report node reads it via ``get_store()``. Without one, memory falls back to
    the on-disk file (used by tests).
    """
    g = StateGraph(AgentState)

    g.add_node("guardrail", guardrail)
    g.add_node("retrieve_trios", retrieve_trios)
    g.add_node("generate_sql", generate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("mask_pii", mask_pii)
    g.add_node("report", report)

    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", _after_guardrail, ["retrieve_trios", END])
    g.add_edge("retrieve_trios", "generate_sql")
    g.add_edge("generate_sql", "execute_sql")
    g.add_conditional_edges("execute_sql", _after_execute, ["generate_sql", "mask_pii"])
    g.add_edge("mask_pii", "report")
    g.add_edge("report", END)

    return g.compile(store=store)
