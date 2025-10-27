"""Module for embeddings_sparse.base.

NavMap:
- NavMap: Structure describing a module navmap.
- SparseEncoder: Protocol for sparse text encoders.
- SparseIndex: Protocol describing sparse index interactions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final, Protocol

from kgfoundry_common.navmap_types import NavMap

__all__ = ["SparseEncoder", "SparseIndex"]

__navmap__: Final[NavMap] = {
    "title": "embeddings_sparse.base",
    "synopsis": "Module for embeddings_sparse.base",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["SparseEncoder", "SparseIndex"],
        },
    ],
}


# [nav:anchor SparseEncoder]
class SparseEncoder(Protocol):
    """Protocol for sparse text encoders."""

    name: str

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Return sparse encodings for the given texts."""
        ...


# [nav:anchor SparseIndex]
class SparseIndex(Protocol):
    """Protocol describing sparse index interactions."""

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Build the index from the supplied documents."""
        ...

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Search the index and return (id, score) tuples."""
        ...
