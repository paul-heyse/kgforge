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
    """Protocol describing sparse encoders exposed by the package.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    name : str
        Canonical identifier advertised by the encoder implementation.
    """

    name: str

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Compute encode.
<!-- auto:docstring-builder v1 -->

Carry out the encode operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
texts : list[str]
    Description for ``texts``.
    
    
    

Returns
-------
list[tuple[list[int], list[float]]]
    Description of return value.
    
    
    

Examples
--------
>>> from embeddings_sparse.base import encode
>>> result = encode(...)
>>> result  # doctest: +ELLIPSIS
"""
        ...


# [nav:anchor SparseIndex]
class SparseIndex(Protocol):
    """Model the SparseIndex.
<!-- auto:docstring-builder v1 -->

Represent the sparseindex data structure used throughout the project. The class encapsulates
behaviour behind a well-defined interface for collaborating components. Instances are typically
created by factories or runtime orchestrators documented nearby.

Parameters
----------
*args : Any
    Describe ``args``.
**kwargs : Any
    Describe ``kwargs``.
"""

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Compute build.
<!-- auto:docstring-builder v1 -->

Carry out the build operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
docs_iterable : Iterable[tuple[str, dict[str, str]]]
    Description for ``docs_iterable``.
    
    
    

Examples
--------
>>> from embeddings_sparse.base import build
>>> build(...)  # doctest: +ELLIPSIS
"""
        ...

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Compute search.
<!-- auto:docstring-builder v1 -->

Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
query : str
    Description for ``query``.
k : int
    Description for ``k``.
fields : Mapping[str, str] | None, optional
    Defaults to ``None``.
    Description for ``fields``.
    
    
    
    Defaults to ``None``.

Returns
-------
list[tuple[str, float]]
    Description of return value.
    
    
    

Examples
--------
>>> from embeddings_sparse.base import search
>>> result = search(..., ...)
>>> result  # doctest: +ELLIPSIS
"""
        ...
