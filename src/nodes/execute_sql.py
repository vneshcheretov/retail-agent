"""Execute SQL node: run the query and capture rows or a clean error.

The graph decides whether to retry; this node only runs the query and records
what happened (error, empty result, or rows).
"""

from __future__ import annotations

from src.services.bigquery_client import QueryError, run_query, validate_query
from src.state import AgentState


def execute_sql(state: AgentState) -> AgentState:
    """Run the current SQL and store rows, or an error message for self-correction.

    First a free BigQuery dry run validates syntax and checks the byte estimate,
    so a broken or too-expensive query is caught (and self-corrected) before the
    real, billed run.
    """
    obs = state["observer"]
    with obs.node("execute_sql"):
        obs.metrics.sql_attempts += 1

        # Free pre-validation: catch syntax errors and over-budget queries.
        try:
            scanned = validate_query(state["sql"])
            obs.event("execute_sql", stage="dry_run", scanned_bytes=scanned)
        except QueryError as exc:
            obs.metrics.sql_errors += 1
            obs.event("execute_sql", stage="dry_run", ok=False, error=str(exc))
            return {"rows": [], "error": str(exc)}

        try:
            rows = run_query(state["sql"])
        except QueryError as exc:
            obs.metrics.sql_errors += 1
            obs.event("execute_sql", ok=False, error=str(exc))
            return {"rows": [], "error": str(exc)}

        if not rows:
            obs.metrics.empty_results += 1
            obs.event("execute_sql", ok=True, row_count=0)
            return {"rows": [], "error": "Query returned no rows."}

        obs.event("execute_sql", ok=True, row_count=len(rows))
        return {"rows": rows, "error": None}
