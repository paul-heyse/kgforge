"""Overview of splade index.

This module bundles splade index logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
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
    "synopsis": "Example SPLADE index used in the search API fixtures",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}

TOKEN = re.compile(r"[A-Za-z0-9]+")


# [nav:anchor tok]
def tok(text: str) -> list[str]:
    """Compute tok.
<!-- auto:docstring-builder v1 -->

Carry out the tok operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
text : str
    Description for ``text``.
    
    
    

Returns
-------
list[str]
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.splade_index import tok
>>> result = tok(...)
>>> result  # doctest: +ELLIPSIS
"""
    return [token.lower() for token in TOKEN.findall(text or "")]


# [nav:anchor SpladeDoc]
@dataclass
class SpladeDoc:
    """Fixture SPLADE document loaded from the DuckDB catalog.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    chunk_id : str
        Unique chunk identifier.
    doc_id : str
        Parent document identifier.
    section : str
        Section heading captured with the chunk.
    text : str
        Raw text body of the chunk.
    """

    chunk_id: str
    doc_id: str
    section: str
    text: str


# [nav:anchor SpladeIndex]
class SpladeIndex:
    """Simplified SPLADE index for fixtures and tutorials.
<!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    db_path : str
        DuckDB database containing the fixture datasets.
    chunks_dataset_root : str | None, optional
        Optional explicit path to the chunks dataset. Defaults to ``None`` to
        auto-discover via DuckDB metadata.
    sparse_root : str | None, optional
        Reserved parameter for compatibility with production code paths.
    """

    def __init__(
        self,
        db_path: str,
        chunks_dataset_root: str | None = None,
        sparse_root: str | None = None,
    ) -> None:
        _ = sparse_root  # retained for interface compatibility
        self.db_path = db_path
        self.docs: list[SpladeDoc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self._load(chunks_dataset_root)

    def _load(self, chunks_root: str | None) -> None:
        """Compute load.
<!-- auto:docstring-builder v1 -->

Carry out the load operation.

Parameters
----------
chunks_root : str | None
    Description for ``chunks_root``.
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
        """Compute search.
<!-- auto:docstring-builder v1 -->

Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
query : str
    Description for ``query``.
k : int, optional
    Defaults to ``10``.
    Description for ``k``.
    
    
    
    Defaults to ``10``.

Returns
-------
list[tuple[int, float]]
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.splade_index import search
>>> result = search(...)
>>> result  # doctest: +ELLIPSIS
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
        """Compute doc.
<!-- auto:docstring-builder v1 -->

Carry out the doc operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
"""
        return self.docs[index]
