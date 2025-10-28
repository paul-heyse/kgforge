"""Bm25 Index utilities."""

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
            "symbols": ["toks", "BM25Doc", "BM25Index"],
        },
    ],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# [nav:anchor toks]
def toks(text: str) -> list[str]:
    """Compute toks.

    Carry out the toks operation.

    Parameters
    ----------
    text : str
        Description for ``text``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


# [nav:anchor BM25Doc]
@dataclass
class BM25Doc:
    """Describe BM25Doc."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    tf: dict[str, float]
    dl: float


# [nav:anchor BM25Index]
class BM25Index:
    """Describe BM25Index."""

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        k1 : float | None
            Description for ``k1``.
        b : float | None
            Description for ``b``.
        """
        
        
        
        
        
        
        
        
        
        self.k1 = k1
        self.b = b
        self.docs: list[BM25Doc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self.avgdl = 0.0

    @classmethod
    def build_from_duckdb(cls, db_path: str) -> BM25Index:
        """Compute build from duckdb.

        Carry out the build from duckdb operation.

        Parameters
        ----------
        db_path : str
            Description for ``db_path``.

        Returns
        -------
        BM25Index
            Description of return value.
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

        Carry out the build operation.

        Parameters
        ----------
        rows : Iterable[Tuple[str, str, str, str, str]]
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

    def save(self, path: str) -> None:
        """Compute save.

        Carry out the save operation.

        Parameters
        ----------
        path : str
            Description for ``path``.
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

        Carry out the load operation.

        Parameters
        ----------
        path : str
            Description for ``path``.

        Returns
        -------
        BM25Index
            Description of return value.
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

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Compute search.

        Carry out the search operation.

        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int | None
            Description for ``k``.

        Returns
        -------
        List[Tuple[int, float]]
            Description of return value.
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
        return [(index, score) for index, score in ranked[:k] if score > 0.0]

    def doc(self, index: int) -> BM25Doc:
        """Compute doc.

        Carry out the doc operation.

        Parameters
        ----------
        index : int
            Description for ``index``.

        Returns
        -------
        BM25Doc
            Description of return value.
        """
        
        
        
        
        
        
        
        
        
        return self.docs[index]
