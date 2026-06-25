"""Generate real Trios (Question -> SQL -> Report) for the golden bucket.

For each question we: ask the LLM for BigQuery SQL, run it on the real dataset,
mask PII in the rows, then ask the LLM for a short analyst report grounded in the
real numbers. This produces a golden dataset with genuine figures, not made-up
ones. The run is sequential to stay gentle on rate limits.

Usage:
    python scripts/generate_trios.py            # target 100 trios
    python scripts/generate_trios.py 30         # custom target
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.mask_pii import mask_rows  # noqa: E402
from src.services.bigquery_client import QueryError, get_schema, run_query  # noqa: E402
from src.services.llm import get_llm  # noqa: E402
from src.settings import settings  # noqa: E402

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Curated, diverse questions answerable with the four allowed tables.
QUESTION_BANK: list[str] = [
    # Revenue & sales totals
    "What is our total revenue all time?",
    "What is the total revenue for the current year?",
    "What is our monthly revenue this year?",
    "What is the revenue for each calendar year?",
    "What is the average order value?",
    "How many orders have we had in total?",
    "How many orders were placed each month this year?",
    "What is our revenue by quarter for the last two years?",
    "What is the total number of items sold all time?",
    "What is the average number of items per order?",
    # Product performance
    "Which product categories bring the most revenue?",
    "What are the 10 best selling products by units?",
    "Which brands generate the most revenue?",
    "How much revenue does each department make?",
    "Which 10 products have the highest profit margin?",
    "What is the average retail price by category?",
    "Which categories have the most products in the catalog?",
    "What are the top 10 products by revenue?",
    "Which brands have the most products?",
    "What is the average product cost by department?",
    "Which products have the largest gap between retail price and cost?",
    "What is total profit by category?",
    # Customer behavior
    "Who are our top 5 customers by total spend?",
    "Who are our top 10 customers by number of orders?",
    "How much does the average customer spend?",
    "How many customers have placed more than one order?",
    "What share of customers are repeat buyers?",
    "What is the average number of orders per customer?",
    "Which customers spent the most in the current year?",
    "How many one-time customers do we have?",
    # Returns & cancellations
    "Which product categories have the highest return rate?",
    "What is our overall return rate?",
    "How many orders were cancelled in total?",
    "What is the cancellation rate by month this year?",
    "Which brands have the highest return rate?",
    "How much revenue did we lose to returns?",
    "What is the return rate by department?",
    # Geography
    "How many customers do we have per country? Top 10.",
    "Which countries generate the most revenue?",
    "What are the top 10 US states by number of customers?",
    "Which cities have the most customers? Top 10.",
    "What is the average order value by country?",
    "How many orders come from each country? Top 10.",
    # Time-based / trends
    "What is revenue by day of week?",
    "Which month of the year has the highest revenue on average?",
    "How has the number of new customers changed month over month this year?",
    "What is the revenue trend by quarter?",
    "Which weekday has the most orders?",
    "How many orders were delivered each month this year?",
    # Demographics
    "What is the age distribution of our customers?",
    "What is the average age of our customers?",
    "How does revenue split between male and female customers?",
    "What is the average spend by gender?",
    "How many customers are in each age group (under 25, 25-40, 40-60, 60+)?",
    "Which gender places more orders?",
    # Acquisition
    "How many customers come from each traffic source?",
    "Which traffic source brings the highest spending customers?",
    "What is the revenue by traffic source?",
    "What share of customers come from each traffic source?",
    # Fulfilment timing
    "What is the average time from order to delivery in days?",
    "What is the average shipping time in days by month this year?",
    "How many orders are still not delivered?",
    "What is the distribution of order statuses?",
    "What is the average time from order to shipment?",
]


def _llm_with_retry(prompt, attempts: int = 3):
    """Call the LLM, retrying a few times on transient errors."""
    llm = get_llm()
    for i in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - retry transient API/rate errors
            if i == attempts - 1:
                raise
            print(f"    LLM error ({exc}); retry {i + 1}")
            time.sleep(3 * (i + 1))


def _clean_sql(text: str) -> str:
    """Strip markdown fences the model sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip().strip("`").strip()


def expand_questions(pool: list[str], target_pool: int) -> list[str]:
    """Top up the question bank with extra distinct questions via the LLM."""
    if len(pool) >= target_pool:
        return pool
    need = target_pool - len(pool)
    prompt = [
        (
            "system",
            "You generate analytics questions a retail manager would ask. "
            "Use only these tables: orders, order_items, products, users from the "
            "thelook_ecommerce dataset. Keep questions answerable by SQL.",
        ),
        (
            "human",
            f"Give {need} short, distinct questions, one per line, no numbering. "
            f"Avoid repeating these:\n" + "\n".join(pool),
        ),
    ]
    response = _llm_with_retry(prompt)
    extra = [line.strip("-• ").strip() for line in response.content.splitlines() if line.strip()]
    seen = {q.lower() for q in pool}
    for q in extra:
        if q.lower() not in seen and "?" in q:
            pool.append(q)
            seen.add(q.lower())
    return pool


def generate_sql(question: str, error: str | None = None, prev_sql: str | None = None) -> str:
    """Ask the LLM for a BigQuery query (or a corrected one after an error)."""
    system = (
        "You write BigQuery Standard SQL for the thelook_ecommerce dataset.\n"
        "Rules: SELECT only; fully-qualify tables like "
        "`bigquery-public-data.thelook_ecommerce.orders`; add LIMIT <= 100 unless "
        "the question needs an aggregate total; return only SQL, no markdown.\n\n"
        f"Schema:\n{get_schema()}"
    )
    prompt = [("system", system), ("human", f"Question: {question}")]
    if error:
        prompt.append(("human", f"Previous SQL failed:\n{prev_sql}\nError: {error}\nFix it."))
    return _clean_sql(_llm_with_retry(prompt).content)


def generate_report(question: str, rows: list[dict]) -> str:
    """Ask the LLM for a short analyst report grounded in the real rows."""
    prompt = [
        (
            "system",
            "You are a retail data analyst writing for a non-technical manager. "
            "Base the report only on the data rows. Use the real numbers. Keep it "
            "under 110 words, lead with the answer, end with one 'So what' line. "
            "Never include customer emails, phones, or addresses.",
        ),
        (
            "human",
            f"Question: {question}\n\nData rows (JSON):\n{json.dumps(rows[:30], default=str)}",
        ),
    ]
    return _llm_with_retry(prompt).content.strip()


def build_trio(question: str) -> dict | None:
    """Run the full pipeline for one question. Returns a trio or None on failure."""
    sql = generate_sql(question)
    try:
        rows = run_query(sql)
    except QueryError as exc:
        sql = generate_sql(question, error=str(exc), prev_sql=sql)  # one self-correction
        try:
            rows = run_query(sql)
        except QueryError as exc2:
            print(f"    skip (SQL failed twice): {exc2}")
            return None
    if not rows:
        print("    skip (empty result)")
        return None
    masked, _ = mask_rows(rows)
    report = generate_report(question, masked)
    return {"question": question, "sql": sql, "report": report}


def main() -> None:
    out_path = settings.seed_trios_path
    backup = out_path.with_suffix(".handwritten.json")
    if out_path.exists() and not backup.exists():
        backup.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backed up existing trios to {backup.name}")

    pool = expand_questions(list(QUESTION_BANK), target_pool=int(TARGET * 1.4))
    print(f"Question pool: {len(pool)} candidates, target {TARGET} trios\n")

    trios: list[dict] = []
    for i, question in enumerate(pool, 1):
        if len(trios) >= TARGET:
            break
        print(f"[{len(trios)}/{TARGET}] ({i}) {question}")
        trio = build_trio(question)
        if trio:
            trios.append(trio)
            if len(trios) % 10 == 0:
                out_path.write_text(json.dumps(trios, indent=2), encoding="utf-8")
                print(f"    ...saved progress ({len(trios)})")
        time.sleep(0.4)  # be gentle on rate limits

    out_path.write_text(json.dumps(trios, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {len(trios)} trios to {out_path}")
    if len(trios) < TARGET:
        print(f"Note: reached {len(trios)}/{TARGET}; rerun to add more.")


if __name__ == "__main__":
    main()
