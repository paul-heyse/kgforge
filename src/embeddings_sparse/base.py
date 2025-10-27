"""Module for embeddings_sparse.base."""

from __future__ import annotations

from typing import Dict, Iterable, List, Protocol, Tuple


class SparseEncoder(Protocol):
    """Protocol for sparse text encoders."""

    name: str

    def encode(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        """Return sparse encodings for the given texts."""

        ...


class SparseIndex(Protocol):
    """Protocol describing sparse index interactions."""

    def build(self, docs_iterable: Iterable[Tuple[str, Dict]]) -> None:
        """Build the index from the supplied documents."""

        ...

    def search(self, query: str, k: int, fields: Dict | None = None) -> List[Tuple[str, float]]:
        """Search the index and return (id, score) tuples."""

        ...
