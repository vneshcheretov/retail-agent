"""Seed the golden bucket with the hand-written Trios in src/data/seed_trios.json.

Run once before the first chat session:
    python scripts/seed_golden_bucket.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.golden_bucket import GoldenBucket, Trio  # noqa: E402
from src.settings import settings  # noqa: E402


def main() -> None:
    trios_data = json.loads(settings.seed_trios_path.read_text(encoding="utf-8"))
    bucket = GoldenBucket()
    for item in trios_data:
        bucket.add_trio(Trio(question=item["question"], sql=item["sql"], report=item["report"]))
    print(f"Seeded {len(trios_data)} trios. Bucket now holds {bucket.count()}.")


if __name__ == "__main__":
    main()
