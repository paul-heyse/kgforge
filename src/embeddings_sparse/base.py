"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
embeddings_sparse.base
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
    """
    Represent SparseEncoder.
    
    Attributes
    ----------
    name : str
        Attribute description.
    
    Methods
    -------
    encode()
        Method description.
    
    Examples
    --------
    >>> from embeddings_sparse.base import SparseEncoder
    >>> result = SparseEncoder()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    embeddings_sparse.base
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    

    name: str

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """
        Return encode.
        
        Parameters
        ----------
        texts : List[str]
            Description for ``texts``.
        
        Returns
        -------
        List[Tuple[List[int], List[float]]]
            Description of return value.
        
        Examples
        --------
        >>> from embeddings_sparse.base import encode
        >>> result = encode(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        embeddings_sparse.base
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        ...


# [nav:anchor SparseIndex]
class SparseIndex(Protocol):
    """
    Represent SparseIndex.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    build()
        Method description.
    search()
        Method description.
    
    Examples
    --------
    >>> from embeddings_sparse.base import SparseIndex
    >>> result = SparseIndex()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    embeddings_sparse.base
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """
        Return build.
        
        Parameters
        ----------
        docs_iterable : Iterable[Tuple[str, dict[str, str]]]
            Description for ``docs_iterable``.
        
        Examples
        --------
        >>> from embeddings_sparse.base import build
        >>> build(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        embeddings_sparse.base
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        ...

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """
        Return search.
        
        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int
            Description for ``k``.
        fields : Mapping[str, str] | None, optional
            Description for ``fields``.
        
        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.
        
        Examples
        --------
        >>> from embeddings_sparse.base import search
        >>> result = search(..., ..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        embeddings_sparse.base
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        ...
