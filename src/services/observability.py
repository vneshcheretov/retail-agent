"""Observability: structured per-node logs, run metrics and a console summary.

Every turn creates one ``RunObserver``. Each node logs a JSON line with timing
and any custom fields, and bumps counters on the shared metrics object. At the
end of the turn the CLI prints a short summary built from these metrics.

LangSmith tracing is optional: if ``LANGSMITH_TRACING=true`` and a key is set,
LangChain picks it up from the environment automatically and we do nothing
special here. With no key we simply fall back to the local JSONL logs.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.settings import settings


@dataclass
class RunMetrics:
    """Counters collected over a single turn."""

    run_id: str
    total_ms: float = 0.0
    node_ms: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    sql_attempts: int = 0
    sql_errors: int = 0
    empty_results: int = 0
    pii_redactions: int = 0
    guardrail_blocked: bool = False
    status: str = "ok"


class RunObserver:
    """Collects logs and metrics for one turn and writes them as JSONL."""

    def __init__(self, run_id: str | None = None, verbose: bool = False) -> None:
        self.metrics = RunMetrics(run_id=run_id or uuid.uuid4().hex[:8])
        self.verbose = verbose  # echo each step to the console (dev/debug)
        self._start = time.perf_counter()
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        self._path: Path = settings.log_dir / f"run_{self.metrics.run_id}.jsonl"
        self._log = logging.getLogger("agent")

    def event(self, node: str, **fields: Any) -> None:
        """Append one structured log line for ``node`` (and echo it if verbose)."""
        record = {"run_id": self.metrics.run_id, "node": node, **fields}
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        self._log.debug("%s %s", node, fields)
        # Echo meaningful steps in flow; skip pure timing lines and the final dump.
        if self.verbose and node != "run" and set(fields) != {"latency_ms"}:
            payload = {k: v for k, v in fields.items() if k != "latency_ms"}
            print(f"  │ [{node}] {payload}", flush=True)

    def trace(self, label: str, text: str) -> None:
        """Print a large piece of context (e.g. a prompt) only in verbose mode."""
        if self.verbose:
            print(f"  │ [{label}]\n{text}\n", flush=True)

    @contextmanager
    def node(self, name: str) -> Iterator["RunObserver"]:
        """Time a node and log its duration, even if it raises."""
        start = time.perf_counter()
        try:
            yield self
        finally:
            ms = (time.perf_counter() - start) * 1000
            self.metrics.node_ms[name] = round(ms, 1)
            self.event(name, latency_ms=round(ms, 1))

    def record_llm(self, response: Any) -> None:
        """Add token usage from a LangChain response to the metrics."""
        self.metrics.llm_calls += 1
        usage = getattr(response, "usage_metadata", None) or {}
        self.metrics.input_tokens += usage.get("input_tokens", 0)
        self.metrics.output_tokens += usage.get("output_tokens", 0)

    def finish(self, status: str = "ok") -> RunMetrics:
        """Stamp total time and status, then return the metrics."""
        self.metrics.total_ms = round((time.perf_counter() - self._start) * 1000, 1)
        self.metrics.status = status
        self.event("run", **asdict(self.metrics))
        return self.metrics
