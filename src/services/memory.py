"""Per-user long-term memory, on top of a LangGraph Store.

Memory is a small **structured profile** per user (one ``answer_style`` line and a
deduplicated ``interests`` list), consolidated by a reflection step rather than
appended to. It lives in a LangGraph ``InMemoryStore`` (the framework's native
long-term memory, accessible inside graph nodes), which we hydrate from and
persist to ``user_memory.json`` so it survives restarts. In production the store
would be a persistent backend (e.g. PostgresStore) and the file would go away.
"""

from __future__ import annotations

import json

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from src.services.llm import get_llm
from src.settings import settings

MAX_INTERESTS = 5
NAMESPACE = ("memory",)  # store namespace; key = user id, value = profile dict

_SYSTEM = """You maintain a compact profile of one user for a retail analytics
assistant. Given the EXISTING profile and the RECENT conversation, output a
CONSOLIDATED profile — do not just append to the old one.

- reasoning: briefly note what you kept, merged, dropped, or added, and why.
- answer_style: ONE short line on how they like answers (format: tables/bullets/
  prose; length: concise/detailed; tone). Empty string if not clear yet.
- interests: a DEDUPLICATED list (max 5) of recurring analytical topics they care
  about (e.g. returns, a region, profitability). Only durable patterns seen more
  than once or clearly emphasized — never one-off questions.

Keep it general and high-quality: prefer fewer, sharper items; drop anything
stale or redundant. Never store PII or specific data values.

Example:
  Existing profile: {'answer_style': 'Concise.', 'interests': ['returns']}
  Conversation: user asks for returns by region, in bullet points.
  -> answer_style: "Concise; bullet points."
     interests: ["returns by region"]   # merged region INTO the returns item, no duplicate
     reasoning: "Refined the returns interest with the region angle; added bullets to style."
"""


class UserMemory(BaseModel):
    """Consolidated profile for one user."""

    reasoning: str = Field(default="", description="What changed and why (not shown to the user)")
    answer_style: str = Field(default="", description="One short line on preferred format/length/tone")
    interests: list[str] = Field(default_factory=list, description="Recurring interests, deduplicated, max 5")


# --- file persistence (so the in-memory store survives restarts) -------------


def _read() -> dict:
    if settings.user_memory_path.exists():
        return json.loads(settings.user_memory_path.read_text(encoding="utf-8"))
    return {}


def _persist(store: BaseStore) -> None:
    data = {item.key: item.value for item in store.search(NAMESPACE, limit=1000)}
    settings.user_memory_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_store() -> InMemoryStore:
    """A LangGraph store hydrated from disk; pass it to ``build_graph(store=...)``."""
    store = InMemoryStore()
    for user, profile in _read().items():
        if isinstance(profile, dict):
            store.put(NAMESPACE, user, profile)
    return store


# --- read / render -----------------------------------------------------------


def read_profile(store: BaseStore, user: str) -> dict:
    """Return a user's profile from the store ({} if none)."""
    item = store.get(NAMESPACE, user)
    return item.value if item else {}


def load_memory(user: str) -> dict:
    """Return a user's profile straight from disk (for the CLI / notebooks)."""
    profile = _read().get(user)
    return profile if isinstance(profile, dict) else {}


def render_memory(profile: dict) -> str:
    """Render a profile as a prompt block, or '' if empty."""
    style = (profile.get("answer_style") or "").strip()
    interests = profile.get("interests") or []
    if not style and not interests:
        return ""
    lines = []
    if style:
        lines.append(f"- Preferred answer style: {style}")
    if interests:
        lines.append(f"- Recurring interests: {', '.join(interests)}")
    return "What we know about this user (adapt the report to it):\n" + "\n".join(lines)


def memory_hint(user: str) -> str:
    """Convenience: render a user's on-disk profile (CLI / notebooks)."""
    return render_memory(load_memory(user))


# --- write -------------------------------------------------------------------


def clear_memory(store: BaseStore, user: str) -> None:
    """Forget everything stored about ``user``."""
    store.delete(NAMESPACE, user)
    _persist(store)


def reflect(store: BaseStore, user: str, history: list[dict[str, str]]) -> dict:
    """Consolidate the user's profile from the recent conversation. Returns it."""
    if not history:
        return read_profile(store, user)

    existing = read_profile(store, user)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-12:])
    prompt = [
        ("system", _SYSTEM),
        ("human", f"Existing profile:\n{existing}\n\nRecent conversation:\n{convo}\n\nConsolidated profile:"),
    ]
    result: UserMemory = get_llm().with_structured_output(UserMemory).invoke(prompt)
    profile = {"answer_style": result.answer_style.strip(), "interests": result.interests[:MAX_INTERESTS]}

    store.put(NAMESPACE, user, profile)
    _persist(store)
    return profile
