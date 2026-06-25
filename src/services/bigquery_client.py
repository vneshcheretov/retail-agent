"""Read-only BigQuery access for the agent.

Thin adapter over the project's ``BigQueryRunner`` (bq_client.py). It returns
plain ``list[dict]`` rows (so masking and reporting stay simple), enforces a byte
cap, and turns any failure into a clean ``QueryError`` the agent can self-correct
from. The table schema is read live and cached, with a static fallback so SQL
generation still works if metadata is unavailable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from bq_client import BigQueryRunner

from src.settings import DATASET, TABLES, settings

# Columns that are PII (kept in sync with nodes/mask_pii.py PII_COLUMNS).
_PII_COLUMNS = {"email", "phone", "first_name", "last_name", "street_address"}

# Used only if live schema introspection fails (e.g. no BigQuery credentials).
_STATIC_SCHEMA = f"""
Dataset: `{DATASET}` (read-only)

Table `orders`: order_id, user_id, status, gender, created_at, returned_at,
  shipped_at, delivered_at, num_of_item
Table `order_items`: id, order_id, user_id, product_id, inventory_item_id,
  status, created_at, shipped_at, delivered_at, returned_at, sale_price
Table `products`: id, cost, category, name, brand, retail_price, department,
  sku, distribution_center_id
Table `users`: id, first_name [PII], last_name [PII], email [PII], age, gender,
  state, street_address [PII], postal_code, city, country, latitude, longitude,
  traffic_source, created_at

Always reference tables with the fully qualified name, e.g. `{DATASET}.orders`.
""".strip()


class QueryError(RuntimeError):
    """Raised when a query fails (syntax, permissions, byte cap, etc.)."""


@lru_cache(maxsize=1)
def _runner() -> BigQueryRunner:
    return BigQueryRunner(project_id=settings.gcp_project, dataset_id=DATASET)


def validate_query(sql: str) -> int:
    """Dry-run a query: check syntax and the byte estimate, without running it.

    Returns the estimated bytes scanned. Raises ``QueryError`` on a syntax error
    or if the query would scan more than the cap, so a bad or too-expensive query
    is caught for free before the real (billed) run.
    """
    try:
        scanned = _runner().dry_run(sql)
    except Exception as exc:  # noqa: BLE001 - surface a clean message for self-correction
        raise QueryError(str(exc)) from exc
    if scanned > settings.max_bytes_billed:
        mb, cap = scanned / 1024 / 1024, settings.max_bytes_billed / 1024 / 1024
        raise QueryError(f"Query would scan {mb:.0f} MB, over the {cap:.0f} MB limit. Narrow it down.")
    return scanned


def run_query(sql: str) -> list[dict[str, Any]]:
    """Run a read-only query and return rows as dicts (PII included, unmasked)."""
    try:
        df = _runner().execute_query(sql, max_bytes_billed=settings.max_bytes_billed)
        return df.to_dict("records")
    except Exception as exc:  # noqa: BLE001 - surface a clean message for self-correction
        raise QueryError(str(exc)) from exc


def _format_column(col: dict[str, Any]) -> str:
    """Render one column as 'name type', tagging PII columns."""
    tag = " [PII]" if col["name"].lower() in _PII_COLUMNS else ""
    return f"{col['name']} {col['type']}{tag}"


# --- Disk cache for schema/enum context ------------------------------------
# Profiling enums means many BigQuery round-trips (~1 min). We cache the result
# on disk so it is paid once, not on every process start. Delete the cache file
# (or call clear_context_cache) to refresh after the data/schema changes.


def _load_cache() -> dict:
    if settings.bq_cache_path.exists():
        try:
            return json.loads(settings.bq_cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt cache should just be rebuilt
            return {}
    return {}


def _cache_put(key: str, value: str) -> None:
    data = _load_cache()
    data[key] = value
    settings.bq_cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def context_cached() -> bool:
    """True if schema and enums are already on disk (so startup is instant)."""
    data = _load_cache()
    return "schema" in data and "enums" in data


def clear_context_cache() -> None:
    """Delete the cached schema/enums so they are rebuilt from BigQuery next time."""
    settings.bq_cache_path.unlink(missing_ok=True)
    get_schema.cache_clear()
    get_enums.cache_clear()


@lru_cache(maxsize=1)
def get_schema() -> str:
    """Return a text schema for the prompt (disk-cached; static fallback)."""
    cached = _load_cache().get("schema")
    if cached is not None:
        return cached
    try:
        runner = _runner()
        lines = [f"Dataset: `{DATASET}` (read-only)", ""]
        for table in TABLES:
            cols = runner.get_table_schema(table)
            rendered = ", ".join(_format_column(c) for c in cols)
            lines.append(f"Table `{table}`: {rendered}")
        lines.append("")
        lines.append(f"Always fully-qualify tables, e.g. `{DATASET}.orders`.")
        result = "\n".join(lines)
    except Exception:  # noqa: BLE001 - metadata is optional; static schema is enough
        return _STATIC_SCHEMA  # fallback is not cached
    _cache_put("schema", result)
    return result


# Cardinality thresholds for treating a string column as an enum.
_ENUM_MAX = 10        # <= this: list every value
_EXAMPLES_MAX = 35    # <= this: show top 5 as examples; above: skip the column


@lru_cache(maxsize=1)
def get_enums() -> str:
    """Profile low-cardinality string columns and list their values, from BigQuery.

    Disk-cached. PII columns are skipped. Returns '' if BigQuery is unavailable,
    so SQL generation still works without it.
    """
    cached = _load_cache().get("enums")
    if cached is not None:
        return cached

    from src.nodes.mask_pii import is_pii_column

    try:
        runner = _runner()
        lines: list[str] = []
        for table in TABLES:
            for col in runner.get_table_schema(table):
                name = col["name"]
                if col["type"] != "STRING" or is_pii_column(name):
                    continue
                fq = f"`{DATASET}.{table}`"
                total = run_query(f"SELECT APPROX_COUNT_DISTINCT({name}) AS n FROM {fq}")[0]["n"]
                if total > _EXAMPLES_MAX:
                    continue
                top = run_query(
                    f"SELECT {name} AS v, COUNT(*) AS c FROM {fq} "
                    f"WHERE {name} IS NOT NULL GROUP BY v ORDER BY c DESC LIMIT 5"
                )
                values = ", ".join(str(r["v"]) for r in top)
                kind = f"enum, {total} total" if total <= _ENUM_MAX else f"examples; {total} total"
                lines.append(f"- {table}.{name}: {values} ({kind})")
        result = "Column values:\n" + "\n".join(lines) if lines else ""
    except Exception:  # noqa: BLE001 - enums are optional context
        return ""  # fallback is not cached
    _cache_put("enums", result)
    return result
