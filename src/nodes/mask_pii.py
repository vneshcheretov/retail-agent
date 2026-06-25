"""PII masking node.

The SQL may legitimately return PII (e.g. a query touched the users table), but
PII must never reach the final answer. We mask here, on the result rows, before
they are shown or sent to the report LLM. Masking is independent of the SQL, so
it works even if the model selects a PII column by accident.

Three things mark a value as PII:
1. The column name looks like PII (normalized, so ``e-mail``/``email_address``
   match too) -- see ``is_pii_column``.
2. The LLM that wrote the SQL flagged the output column as PII (``extra_pii``),
   which also catches arbitrary aliases like ``SELECT email AS x``.
3. The value itself matches an email/phone regex anywhere in free text.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.state import AgentState

# Keywords (normalized: lowercase, no separators) that mark a column as PII by
# name. Substrings on purpose, so "customer_email" / "contact_phone" match.
# "name" alone is excluded so "product_name"/"category_name" stay visible.
PII_COLUMNS = {
    "email", "mail", "phone", "mobile", "telephone", "fax",
    "firstname", "lastname", "fullname", "middlename",
    "street", "address",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")


def _normalize(name: str) -> str:
    """Lowercase a column name and drop separators (``e-mail`` -> ``email``)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_pii_column(name: str) -> bool:
    """True if the column name looks like personal data."""
    normalized = _normalize(name)
    return any(keyword in normalized for keyword in PII_COLUMNS)


def _mask_text(value: str) -> tuple[str, int]:
    """Mask any email/phone found in free text. Returns (text, redaction_count)."""
    count = 0
    value, n = EMAIL_RE.subn("[redacted-email]", value)
    count += n
    value, n = PHONE_RE.subn("[redacted-phone]", value)
    count += n
    return value, count


def mask_row(row: dict[str, Any], extra_pii: Iterable[str] | None = None) -> tuple[dict[str, Any], int]:
    """Return a masked copy of one row and how many fields were redacted.

    ``extra_pii`` are output column names the SQL author flagged as PII (matched
    after normalization, so aliases are caught too).
    """
    extra = {_normalize(c) for c in (extra_pii or ())}
    masked: dict[str, Any] = {}
    redactions = 0
    for key, value in row.items():
        # Numbers (and None) are never PII. This avoids masking aggregates whose
        # alias happens to contain a PII word, e.g. `email_revenue`.
        if value is None or isinstance(value, (int, float, bool)):
            masked[key] = value
        elif is_pii_column(key) or _normalize(key) in extra:
            masked[key] = "[redacted]"  # PII-named string column
            redactions += 1
        elif isinstance(value, str):
            new_value, n = _mask_text(value)  # PII hiding in free text
            masked[key] = new_value
            redactions += n
        else:
            masked[key] = value
    return masked, redactions


def mask_rows(rows: list[dict[str, Any]], extra_pii: Iterable[str] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Mask a list of rows; returns the masked rows and total redactions."""
    out, total = [], 0
    for row in rows:
        masked, n = mask_row(row, extra_pii=extra_pii)
        out.append(masked)
        total += n
    return out, total


def mask_pii(state: AgentState) -> AgentState:
    """Mask PII in the query rows before they leave the system."""
    obs = state["observer"]
    with obs.node("mask_pii"):
        extra = state.get("pii_columns", [])
        masked, redactions = mask_rows(state.get("rows", []), extra_pii=extra)
        obs.metrics.pii_redactions += redactions
        obs.event("mask_pii", redactions=redactions, rows=len(masked), flagged_columns=list(extra))
        return {"masked_rows": masked}
