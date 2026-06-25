"""Central runtime settings, loaded once from the environment.

Everything that changes between machines or providers lives here, so the rest
of the code never reads ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# BigQuery public dataset used by the assignment.
DATASET = "bigquery-public-data.thelook_ecommerce"
TABLES = ("orders", "order_items", "products", "users")


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one run of the agent."""

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-5.4-mini"))

    gcp_project: str | None = field(default_factory=lambda: os.getenv("GCP_PROJECT"))

    # Hard cap on bytes scanned per query, so a bad SQL cannot burn the budget.
    max_bytes_billed: int = 200 * 1024 * 1024  # 200 MB

    qdrant_path: Path = ROOT / "qdrant_data"
    log_dir: Path = ROOT / "logs"
    # Cached schema + enum context, so we don't re-profile BigQuery on every start.
    bq_cache_path: Path = ROOT / ".bq_context_cache.json"
    personas_path: Path = ROOT / "config" / "personas.yaml"
    user_memory_path: Path = ROOT / "user_memory.json"

    seed_trios_path: Path = ROOT / "src" / "data" / "seed_trios.json"

    def api_key(self) -> str | None:
        """Return the API key matching the active provider."""
        keys = {
            "openai": "OPENAI_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        return os.getenv(keys.get(self.llm_provider, "OPENAI_API_KEY"))


settings = Settings()
