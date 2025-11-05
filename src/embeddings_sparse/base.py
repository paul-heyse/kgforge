"""Overview of base.

This module bundles base logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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
    """Protocol for sparse embedding encoders.

    Defines the interface for encoders that convert text into sparse
    embeddings. Sparse embeddings consist of token indices and weights,
    typically used for learned sparse retrieval (e.g., SPLADE).

    Attributes
    ----------
    name : str
        Encoder name or identifier.
    """

    name: str

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Encode texts into sparse embeddings.

        Converts a list of text strings into sparse embeddings represented
        as (token_indices, weights) tuples.

        Parameters
        ----------
        texts : list[str]
            List of text strings to encode.

        Returns
        -------
        list[tuple[list[int], list[float]]]
            List of sparse embeddings. Each tuple contains:
            - token_indices: List of vocabulary token indices
            - weights: List of weights corresponding to the token indices
        """
        ...


# [nav:anchor SparseIndex]
class SparseIndex(Protocol):
    """Protocol for sparse embedding indexes.

    Defines the interface for indexes that store and search sparse embeddings. Used for sparse
    retrieval operations where documents are represented as sparse vectors (token indices with
    weights).
    """

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Build the index from documents.

        Indexes documents from the provided iterable. Each document is
        represented as a tuple of (doc_id, fields_dict) where fields_dict
        contains field names mapped to text content.

        Parameters
        ----------
        docs_iterable : Iterable[tuple[str, dict[str, str]]]
            Iterable of document tuples. Each tuple contains:
            - doc_id: Document identifier
            - fields_dict: Dictionary mapping field names to text content
        """
        ...

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Search the index for documents matching the query.

        Performs sparse retrieval to find top-k documents matching the query.
        Supports optional field boosting through the fields parameter.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of top results to return.
        fields : Mapping[str, str] | None, optional
            Optional field boost weights. Maps field names to boost values.
            If None, uses default field weights. Defaults to None.

        Returns
        -------
        list[tuple[str, float]]
            List of (doc_id, score) tuples sorted by score descending.
        """
        ...
