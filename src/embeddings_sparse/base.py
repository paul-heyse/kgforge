"""Base utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final, Protocol

from kgfoundry_common.navmap_types import NavMap

__all__ = ["SparseEncoder", "SparseIndex"]

__navmap__: Final[NavMap] = {
    "title": "embeddings_sparse.base",
    "synopsis": "Protocols for sparse embedding encoders and indices",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@embeddings",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        "SparseEncoder": {
            "owner": "@embeddings",
            "stability": "beta",
            "since": "0.1.0",
        },
        "SparseIndex": {
            "owner": "@embeddings",
            "stability": "beta",
            "since": "0.1.0",
        },
    },
}


# [nav:anchor SparseEncoder]
class SparseEncoder(Protocol):
    """Describe SparseEncoder."""

    name: str

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Compute encode.

        Carry out the encode operation.

        Parameters
        ----------
        texts : List[str]
            Description for ``texts``.

        Returns
        -------
        List[Tuple[List[int], List[float]]]
            Description of return value.
        """
        ...


# [nav:anchor SparseIndex]
class SparseIndex(Protocol):
    """Describe SparseIndex."""

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Compute build.

        Carry out the build operation.

        Parameters
        ----------
        docs_iterable : Iterable[Tuple[str, dict[str, str]]]
            Description for ``docs_iterable``.
        """
        ...

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation.

        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int
            Description for ``k``.
        fields : Mapping[str, str] | None
            Description for ``fields``.

        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.
        """
        ...
