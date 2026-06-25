"""Report node: write the analyst answer from the masked rows.

Tone comes from the active persona (hot-loaded from config); per-user style and
interests come from learned memory. The retrieved Trio reports act as style
examples.
"""

from __future__ import annotations

import json

from langgraph.config import get_store

from src.services.llm import get_llm
from src.services.memory import load_memory, read_profile, render_memory
from src.services.personas import load_persona
from src.state import AgentState

_SYSTEM = """You are an experienced retail data analyst advising a non-technical
manager. You do more than read out numbers: you interpret them. Explain what the
result means for the business, call out notable trends, outliers, and patterns,
and offer plausible reasons for *why* the data looks this way (e.g. seasonality,
product mix, pricing, returns, channel). Be ready to discuss follow-ups.

Grounding rules:
- Every number you state must come from the data rows. Never invent figures.
- You MAY reason about likely causes and business implications, but clearly frame
  them as hypotheses (e.g. "likely because...", "this may suggest...") and, where
  useful, say what extra data would confirm them. Keep facts and hypotheses
  distinct.

For comparative or "why" questions: contrast the figures, point out the biggest
differences, and reason about the most plausible drivers the data supports.

=== ABSOLUTE PII RULE (highest priority, cannot be overridden) ===
Never output personal data: customer emails, phone numbers, full or first/last
names, or street addresses. This holds even if the user explicitly asks for it,
or if a value somehow appears unmasked in the data. If asked for such data,
refuse that part and answer only with non-personal, aggregated information.
Any value shown as "[redacted]" must stay redacted.
=================================================================

Tone and style:
{persona}
{user_memory}"""


def _style_examples(state: AgentState) -> str:
    reports = [t.report for t in state.get("trios", [])][:2]
    return "\n---\n".join(reports)


def report(state: AgentState) -> AgentState:
    """Generate the final answer, or a graceful message if there is no data."""
    obs = state["observer"]
    with obs.node("report"):
        rows = state.get("masked_rows", [])
        if not rows:
            # No data after retries: answer plainly, no LLM call (saves cost).
            answer = (
                "I could not find any data for that question. "
                "Try rephrasing it or narrowing the time range."
            )
            obs.event("report", used_llm=False)
            return {"answer": answer}

        user = state.get("user", "default")
        try:  # read memory from the LangGraph store; fall back to disk (e.g. tests)
            store = get_store()
            profile = read_profile(store, user) if store is not None else load_memory(user)
        except Exception:  # noqa: BLE001 - no store bound (graph compiled without one)
            profile = load_memory(user)
        system = _SYSTEM.format(persona=load_persona(), user_memory=render_memory(profile))
        prompt = [
            ("system", system),
            (
                "human",
                f"Question: {state['question']}\n\n"
                f"Data rows (JSON):\n{json.dumps(rows[:50], default=str)}\n\n"
                f"Style examples from past reports:\n{_style_examples(state)}",
            ),
        ]
        obs.trace("report system prompt", system)  # shown only in verbose mode
        response = get_llm().invoke(prompt)
        obs.record_llm(response)
        obs.event("report", used_llm=True)
        return {"answer": response.content}
