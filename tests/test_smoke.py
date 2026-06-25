"""Offline smoke test: the whole graph runs with fake LLM and BigQuery.

No network or API key needed. It proves the nodes are wired correctly and that
PII is masked end-to-end, including the email a query might return.
"""

from __future__ import annotations

import json

import src.nodes.execute_sql as execute_mod
import src.nodes.generate_sql as gen_mod
import src.nodes.guardrail as guard_mod
import src.nodes.report as report_mod
import src.nodes.retrieve as retrieve_mod
from src.graph import build_graph
from src.services.observability import RunObserver
from src.settings import settings


class FakeMsg:
    """Minimal stand-in for a LangChain AIMessage."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 5, "output_tokens": 5}


class FakeStructured:
    def __init__(self, model) -> None:
        self._model = model

    def invoke(self, _messages):
        if self._model is guard_mod.Decision:
            parsed = guard_mod.Decision(intent="analysis", reply="")
        elif self._model is gen_mod.SqlPlan:
            parsed = gen_mod.SqlPlan(sql="SELECT email FROM users", pii_columns=["email"])
        else:
            parsed = None
        return {"raw": FakeMsg(""), "parsed": parsed}


class FakeLLM:
    def __init__(self, reply: str = "") -> None:
        self._reply = reply

    def with_structured_output(self, model, include_raw=False):
        return FakeStructured(model)

    def invoke(self, _messages):
        return FakeMsg(self._reply)


class FakeBucket:
    def retrieve(self, _question, k=3):
        return []


def test_graph_runs_and_masks_pii(monkeypatch):
    monkeypatch.setattr(guard_mod, "get_llm", lambda: FakeLLM(""))
    monkeypatch.setattr(gen_mod, "get_llm", lambda: FakeLLM("SELECT email FROM users"))
    monkeypatch.setattr(gen_mod, "get_schema", lambda: "schema")  # avoid live BQ probe
    monkeypatch.setattr(gen_mod, "get_enums", lambda: "")          # avoid live BQ probe
    monkeypatch.setattr(report_mod, "get_llm", lambda: FakeLLM("Here is your report."))
    monkeypatch.setattr(retrieve_mod, "_get_bucket", lambda: FakeBucket())
    monkeypatch.setattr(execute_mod, "validate_query", lambda _sql: 0)  # skip live dry run
    monkeypatch.setattr(
        execute_mod,
        "run_query",
        lambda _sql: [{"user_id": 1, "email": "x@y.com"}],
    )

    app = build_graph()
    result = app.invoke(
        {"question": "show top customers", "user": "t", "observer": RunObserver(), "attempts": 0}
    )

    assert result["answer"] == "Here is your report."
    assert result["masked_rows"][0]["email"] == "[redacted]"
    assert result["masked_rows"][0]["user_id"] == 1


def test_seed_trios_are_valid():
    trios = json.loads(settings.seed_trios_path.read_text(encoding="utf-8"))
    assert len(trios) >= 5
    for trio in trios:
        assert {"question", "sql", "report"} <= trio.keys()
        assert "SELECT" in trio["sql"].upper()


def test_graph_compiles():
    assert build_graph() is not None
