"""Generate SQL node: turn the question into a BigQuery query.

Uses the table schema and the retrieved Trios as few-shot examples. The same LLM
call also reports which output columns hold PII (``pii_columns``) -- since the
model picks the column aliases, it is the best judge of which ones are personal
data, even for aliases like ``SELECT email AS x``. On a retry, the previous SQL
and its error are added so the model can fix its own mistake.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.services.bigquery_client import get_enums, get_schema
from src.services.llm import get_llm
from src.state import AgentState

_SYSTEM = """You write BigQuery Standard SQL for a retail analytics database.
Rules:
- Use only the tables and columns in the schema below.
- Read-only: SELECT statements only. Never write DML/DDL.
- Always fully-qualify tables, e.g. `bigquery-public-data.thelook_ecommerce.orders`.
- Add a sensible LIMIT (<= 100) unless the question needs an aggregate total.
- Avoid selecting PII columns unless the question truly needs them.
- For comparative or "why" questions (e.g. comparing two segments, regions, or
  categories, or asking why one underperforms): write ONE query that returns the
  relevant metrics side by side for the things compared -- group by the compared
  dimension, or use conditional aggregation -- and include several angles
  (e.g. revenue, order count, average order value, return rate) so the report
  has enough to compare and explain.

Also return `pii_columns`: the list of OUTPUT column names (use the exact alias
you choose in the SELECT) whose values are personal data -- emails, phones, full
names, first/last names, or street addresses. Return an empty list if none.

Schema:
{schema}"""


class SqlPlan(BaseModel):
    """The generated query plus which of its output columns are PII."""

    sql: str = Field(description="The BigQuery SQL query, no markdown fences")
    pii_columns: list[str] = Field(
        default_factory=list,
        description="Output column names whose values are personal data",
    )


def _few_shot(state: AgentState) -> str:
    """Render retrieved Trios as 'question -> SQL' examples."""
    lines = []
    for trio in state.get("trios", []):
        lines.append(f"Q: {trio.question}\nSQL:\n{trio.sql}")
    return "\n\n".join(lines)


def _clean(sql: str) -> str:
    """Strip markdown fences the model sometimes adds."""
    sql = re.sub(r"^```(?:sql)?", "", sql.strip())
    sql = re.sub(r"```$", "", sql.strip())
    return sql.strip()


def generate_sql(state: AgentState) -> AgentState:
    """Produce a SQL query for the question (or fix the previous one)."""
    obs = state["observer"]
    with obs.node("generate_sql"):
        examples = _few_shot(state)
        system = _SYSTEM.format(schema=get_schema())
        if enums := get_enums():  # real column values, profiled live from BigQuery
            system += "\n\n" + enums
        prompt = [
            ("system", system),
            ("human", f"Similar expert examples:\n\n{examples}\n\nQuestion: {state['question']}"),
        ]
        if state.get("error"):
            prompt.append(
                (
                    "human",
                    f"Your previous query failed:\n{state.get('sql', '')}\n\n"
                    f"Error: {state['error']}\nReturn a corrected query.",
                )
            )

        llm = get_llm().with_structured_output(SqlPlan, include_raw=True)
        result = llm.invoke(prompt)
        obs.record_llm(result["raw"])
        plan: SqlPlan = result["parsed"]
        sql = _clean(plan.sql)
        obs.event("generate_sql", sql=sql, pii_columns=plan.pii_columns)
        return {
            "sql": sql,
            "attempts": state.get("attempts", 0) + 1,
            "pii_columns": plan.pii_columns,
        }
