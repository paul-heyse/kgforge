"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
search_api.splade_index
"""


from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from kgfoundry_common.navmap_types import NavMap

__all__ = ["SpladeDoc", "SpladeIndex", "tok"]

__navmap__: Final[NavMap] = {
    "title": "search_api.splade_index",
    "synopsis": "Toy SPLADE-style sparse index for fixture search endpoints.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["tok", "SpladeDoc", "SpladeIndex"],
        },
    ],
}

TOKEN = re.compile(r"[A-Za-z0-9]+")


# [nav:anchor tok]
def tok(text: str) -> list[str]:
    """
    Return tok.
    
    Parameters
    ----------
    text : str
        Description for ``text``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from search_api.splade_index import tok
    >>> result = tok(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_api.splade_index
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    
    return [token.lower() for token in TOKEN.findall(text or "")]


# [nav:anchor SpladeDoc]
@dataclass
class SpladeDoc:
    """
    Represent SpladeDoc.
    
    Attributes
    ----------
    chunk_id : str
        Attribute description.
    doc_id : str
        Attribute description.
    section : str
        Attribute description.
    text : str
        Attribute description.
    
    Examples
    --------
    >>> from search_api.splade_index import SpladeDoc
    >>> result = SpladeDoc()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_api.splade_index
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    chunk_id: str
    doc_id: str
    section: str
    text: str


# [nav:anchor SpladeIndex]
class SpladeIndex:
    """
    Represent SpladeIndex.
    
    Attributes
    ----------
    None
        No public attributes documented.
    
    Methods
    -------
    __init__()
        Method description.
    _load()
        Method description.
    search()
        Method description.
    doc()
        Method description.
    
    Examples
    --------
    >>> from search_api.splade_index import SpladeIndex
    >>> result = SpladeIndex()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    search_api.splade_index
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    

    def __init__(
        self,
        db_path: str,
        chunks_dataset_root: str | None = None,
        sparse_root: str | None = None,
    ) -> None:
        """
        Return init.
        
        Parameters
        ----------
        db_path : str
            Description for ``db_path``.
        chunks_dataset_root : str | None, optional
            Description for ``chunks_dataset_root``.
        sparse_root : str | None, optional
            Description for ``sparse_root``.
        
        Examples
        --------
        >>> from search_api.splade_index import __init__
        >>> __init__(..., ..., ...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        search_api.splade_index
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        _ = sparse_root  # retained for interface compatibility
        self.db_path = db_path
        self.docs: list[SpladeDoc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self._load(chunks_dataset_root)

    def _load(self, chunks_root: str | None) -> None:
        """
        Return load.
        
        Parameters
        ----------
        chunks_root : str | None
            Description for ``chunks_root``.
        
        Examples
        --------
        >>> from search_api.splade_index import _load
        >>> _load(...)  # doctest: +ELLIPSIS
        
        See Also
        --------
        search_api.splade_index
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        _ = chunks_root  # optional override currently unused
        if not Path(self.db_path).exists():
            return
        con = duckdb.connect(self.db_path)
        try:
            dataset = con.execute(
                "SELECT parquet_root FROM datasets "
                "WHERE kind='chunks' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if dataset:
                rows = con.execute(
                    "SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text "
                    f"FROM read_parquet('{dataset[0]}/*/*.parquet', union_by_name=true) AS c"
                ).fetchall()
                for chunk_id, doc_id, section, text in rows:
                    self.docs.append(
                        SpladeDoc(
                            chunk_id=chunk_id,
                            doc_id=doc_id or "urn:doc:fixture",
                            section=section,
                            text=text or "",
                        )
                    )
        finally:
            con.close()
        self.N = len(self.docs)
        for doc in self.docs:
            for term in set(tok(doc.text)):
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """
        Return search.
        
        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int, optional
            Description for ``k``.
        
        Returns
        -------
        List[Tuple[int, float]]
            Description of return value.
        
        Examples
        --------
        >>> from search_api.splade_index import search
        >>> result = search(..., ...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_api.splade_index
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        
        if self.N == 0:
            return []
        terms = tok(query)
        if not terms:
            return []
        scores = [0.0] * self.N
        for index, doc in enumerate(self.docs):
            term_freq: dict[str, int] = {}
            for term in tok(doc.text):
                term_freq[term] = term_freq.get(term, 0) + 1
            score = 0.0
            for term in terms:
                if term in self.df:
                    idf = (self.N + 1) / (self.df[term] + 0.5)
                    score += term_freq.get(term, 0) * idf
            scores[index] = score
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(idx, value) for idx, value in ranked[:k] if value > 0.0]

    def doc(self, index: int) -> SpladeDoc:
        """
        Return doc.
        
        Parameters
        ----------
        index : int
            Description for ``index``.
        
        Returns
        -------
        SpladeDoc
            Description of return value.
        
        Examples
        --------
        >>> from search_api.splade_index import doc
        >>> result = doc(...)
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        search_api.splade_index
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
        """
        
        return self.docs[index]
