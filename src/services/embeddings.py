"""Text embedders for the golden bucket.

A small interface with one implementation (fastembed, fully local, no API key).
Keeping the interface lets us drop in another embedder later (e.g. SONAR) without
touching the rest of the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache


class Embedder(ABC):
    """Turns text into fixed-size vectors for similarity search."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector size produced by this embedder."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


class FastEmbedEmbedder(Embedder):
    """Local embedder based on fastembed (BAAI/bge-small-en-v1.5, 384 dims).

    The model (~50 MB) downloads once on first use and is cached on disk.
    """

    _DIM = 384

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    @property
    def dim(self) -> int:
        return self._DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide embedder instance."""
    return FastEmbedEmbedder()
