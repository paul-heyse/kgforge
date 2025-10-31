"""Overview of base.

This module bundles base logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

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
    """Describe SparseEncoder.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.


    Returns
    -------
    inspect._empty
        Describe return value.
    """

    name: str

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Describe encode.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        texts : list[str]
            Describe ``texts``.


        Returns
        -------
        list[tuple[list[int], list[float]]]
            Describe return value.
        """
        ...


# [nav:anchor SparseIndex]
class SparseIndex(Protocol):
    """Describe SparseIndex.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.


    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Describe build.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        docs_iterable : Iterable[tuple[str, dict[str, str]]]
            Describe ``docs_iterable``.
        """
        ...

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Describe search.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int
            Describe ``k``.
        fields : Mapping[str, str] | None, optional
            Describe ``fields``.
            Defaults to ``None``.


        Returns
        -------
        list[tuple[str, float]]
            Describe return value.
        """
        ...
