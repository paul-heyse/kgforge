"""Overview of bm25 index.

This module bundles bm25 index logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import math
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from kgfoundry_common.navmap_types import NavMap

__all__ = ["BM25Doc", "BM25Index", "toks"]

__navmap__: Final[NavMap] = {
    "title": "search_api.bm25_index",
    "synopsis": "Toy BM25 index backed by DuckDB parquet exports.",
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

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# [nav:anchor toks]
def toks(text: str) -> list[str]:
    """Compute toks.
<!-- auto:docstring-builder v1 -->

Carry out the toks operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

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
>>> from search_api.bm25_index import toks
>>> result = toks(...)
>>> result  # doctest: +ELLIPSIS
"""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


# [nav:anchor BM25Doc]
@dataclass
class BM25Doc:
    """Fixture BM25 document entry.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    chunk_id : str
        Unique chunk identifier.
    doc_id : str
        Document identifier associated with the chunk.
    title : str
        Title string for the document.
    section : str
        Section metadata captured for the chunk.
    tf : dict[str, float]
        Term-frequency mapping used when computing BM25 scores.
    dl : float
        Document length normalisation term.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    tf: dict[str, float]
    dl: float


# [nav:anchor BM25Index]
class BM25Index:
    """Simple BM25 implementation used for fixture data.
<!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    k1 : float, optional
        Saturation parameter controlling term frequency scaling. Defaults to ``0.9``.
    b : float, optional
        Length normalisation parameter. Defaults to ``0.4``.
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:

        self.k1 = k1
        self.b = b
        self.docs: list[BM25Doc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self.avgdl = 0.0

    @classmethod
    def build_from_duckdb(cls, db_path: str) -> BM25Index:
        """Compute build from duckdb.
<!-- auto:docstring-builder v1 -->

Carry out the build from duckdb operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
db_path : str
    Description for ``db_path``.
    
    
    

Returns
-------
BM25Index
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.bm25_index import build_from_duckdb
>>> result = build_from_duckdb(...)
>>> result  # doctest: +ELLIPSIS
"""
        index = cls()
        con = duckdb.connect(db_path)
        try:
            dataset = con.execute(
                "SELECT parquet_root FROM datasets "
                "WHERE kind='chunks' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not dataset:
                return index
            root = dataset[0]
            rows = con.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text, coalesce(d.title,'')
                FROM read_parquet('{root}/*/*.parquet', union_by_name=true) AS c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """
            ).fetchall()
        finally:
            con.close()
        index._build(rows)
        return index

    def _build(self, rows: Iterable[tuple[str, str, str, str, str]]) -> None:
        """Compute build.
<!-- auto:docstring-builder v1 -->

Carry out the build operation.

Parameters
----------
rows : Iterable[tuple[str, str, str, str, str]]
    Description for ``rows``.
"""
        self.docs.clear()
        self.df.clear()
        dl_sum = 0.0
        for chunk_id, doc_id, section, body, title in rows:
            tf: dict[str, float] = {}
            for term in toks(body or ""):
                tf[term] = tf.get(term, 0.0) + 1.0
            for term in toks(title or ""):
                tf[term] = tf.get(term, 0.0) + 2.0
            for term in toks(section or ""):
                tf[term] = tf.get(term, 0.0) + 1.2
            dl = sum(tf.values())
            self.docs.append(
                BM25Doc(
                    chunk_id=chunk_id,
                    doc_id=doc_id or "urn:doc:fixture",
                    title=title or "Fixture",
                    section=section or "",
                    tf=tf,
                    dl=dl,
                )
            )
            dl_sum += dl
            for term in set(tf.keys()):
                self.df[term] = self.df.get(term, 0) + 1
        self.N = len(self.docs)
        self.avgdl = (dl_sum / self.N) if self.N > 0 else 0.0

    @classmethod
    def from_parquet(cls, path: str, *, k1: float = 0.9, b: float = 0.4) -> BM25Index:
        """Instantiate an index from a parquet dataset without DuckDB metadata.
<!-- auto:docstring-builder v1 -->

Parameters
----------
path : str
    Describe ``path``.
k1 : float, optional
    Describe ``k1``.
    Defaults to ``0.9``.
    Defaults to ``0.9``.
b : float, optional
    Describe ``b``.
    
    Defaults to ``0.4``.
    
    Defaults to ``0.4``.

Returns
-------
BM25Index
    Describe return value.
"""
        index = cls(k1=k1, b=b)
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"""
                SELECT chunk_id,
                       coalesce(doc_id, chunk_id) AS doc_id,
                       coalesce(section,'') AS section,
                       text,
                       '' AS title
                FROM read_parquet('{path}', union_by_name=true)
            """
            ).fetchall()
        finally:
            con.close()
        index._build(rows)
        return index

    def save(self, path: str) -> None:
        """Compute save.
<!-- auto:docstring-builder v1 -->

Carry out the save operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
path : str
    Description for ``path``.
    
    
    

Examples
--------
>>> from search_api.bm25_index import save
>>> save(...)  # doctest: +ELLIPSIS
"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "N": self.N,
                    "avgdl": self.avgdl,
                    "df": self.df,
                    "docs": self.docs,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str) -> BM25Index:
        """Compute load.
<!-- auto:docstring-builder v1 -->

Carry out the load operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
path : str
    Description for ``path``.
    
    
    

Returns
-------
BM25Index
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.bm25_index import load
>>> result = load(...)
>>> result  # doctest: +ELLIPSIS
"""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        index = cls(payload.get("k1", 0.9), payload.get("b", 0.4))
        index.N = payload["N"]
        index.avgdl = payload["avgdl"]
        index.df = payload["df"]
        index.docs = payload["docs"]
        return index

    def _idf(self, term: str) -> float:
        """Compute idf.
<!-- auto:docstring-builder v1 -->

Carry out the idf operation.

Parameters
----------
term : str
    Description for ``term``.
    
    
    

Returns
-------
float
    Description of return value.
"""
        df = self.df.get(term, 0)
        if self.N == 0 or df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
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
list[tuple[str, float]]
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.bm25_index import search
>>> result = search(...)
>>> result  # doctest: +ELLIPSIS
"""
        if self.N == 0:
            return []
        terms = toks(query)
        scores = [0.0] * self.N
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.tf.get(term, 0.0)
                if tf <= 0.0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc.dl / (self.avgdl or 1.0)))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            scores[i] = score
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(self.docs[index].chunk_id, score) for index, score in ranked[:k] if score > 0.0]

    def doc(self, index: int) -> BM25Doc:
        """Compute doc.
<!-- auto:docstring-builder v1 -->

Carry out the doc operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
index : int
    Description for ``index``.
    
    
    

Returns
-------
BM25Doc
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.bm25_index import doc
>>> result = doc(...)
>>> result  # doctest: +ELLIPSIS
"""
        return self.docs[index]
