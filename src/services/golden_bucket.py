"""Golden Bucket: a vector store of expert "Trios" (Question -> SQL -> Report).

At query time we retrieve the Trios whose questions are most similar to the
user's question and pass them to the LLM as few-shot examples. This is how the
agent reuses analyst knowledge instead of writing SQL from scratch.

We use Qdrant in embedded mode (a local folder, no server, no Docker), which is
the simplest thing that persists across runs and is easy to run on any machine.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from src.services.embeddings import Embedder, get_embedder
from src.settings import settings

COLLECTION = "golden_trios"


@dataclass
class Trio:
    """One expert example: a question, its SQL, and the analyst report."""

    question: str
    sql: str
    report: str


class GoldenBucket:
    """Stores and retrieves Trios from an embedded Qdrant collection."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or get_embedder()
        self._client = QdrantClient(path=str(settings.qdrant_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection on first use."""
        if self._client.collection_exists(COLLECTION):
            return
        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=self._embedder.dim,
                distance=models.Distance.COSINE,
            ),
        )

    def add_trio(self, trio: Trio) -> None:
        """Embed a Trio's question and store the full Trio as payload."""
        vector = self._embedder.embed([trio.question])[0]
        self._client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={"question": trio.question, "sql": trio.sql, "report": trio.report},
                )
            ],
        )

    def retrieve(self, question: str, k: int = 3) -> list[Trio]:
        """Return the ``k`` Trios most similar to ``question``."""
        vector = self._embedder.embed([question])[0]
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=k,
        ).points
        return [
            Trio(question=h.payload["question"], sql=h.payload["sql"], report=h.payload["report"])
            for h in hits
        ]

    def count(self) -> int:
        """Number of Trios currently stored."""
        return self._client.count(COLLECTION).count
