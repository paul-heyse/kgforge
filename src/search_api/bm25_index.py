"""Module for search_api.bm25_index.

NavMap:
- toks: Toks.
- BM25Doc: Bm25doc.
- BM25Index: Bm25index.
"""

from __future__ import annotations

import math
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def toks(s: str) -> list[str]:
    """Toks.

    Parameters
    ----------
    s : str
        TODO.

    Returns
    -------
    List[str]
        TODO.
    """
    return [t.lower() for t in TOKEN_RE.findall(s or "")]


@dataclass
class BM25Doc:
    """Bm25doc."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    tf: dict[str, float]
    dl: float


class BM25Index:
    """Bm25index."""

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        """Init.

        Parameters
        ----------
        k1 : float
            TODO.
        b : float
            TODO.
        """
        self.k1 = k1
        self.b = b
        self.docs: list[BM25Doc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self.avgdl = 0.0

    @classmethod
    def build_from_duckdb(cls, db_path: str) -> BM25Index:
        """Build from duckdb.

        Parameters
        ----------
        db_path : str
            TODO.

        Returns
        -------
        "BM25Index"
            TODO.
        """
        idx = cls()
        con = duckdb.connect(db_path)
        try:
            ds = con.execute(
                "SELECT parquet_root FROM datasets "
                "WHERE kind='chunks' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not ds:
                return idx
            root = ds[0]
            rows = con.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text, coalesce(d.title,'')
                FROM read_parquet('{root}/*/*.parquet', union_by_name=true) AS c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """
            ).fetchall()
        finally:
            con.close()
        idx._build(rows)
        return idx

    def _build(self, rows: Iterable[tuple[str, str, str, str, str]]) -> None:
        """Build.

        Parameters
        ----------
            rows: TODO.
        """
        self.docs.clear()
        self.df.clear()
        dl_sum = 0.0
        for chunk_id, doc_id, section, body, title in rows:
            tf: dict[str, float] = {}
            for t in toks(body or ""):
                tf[t] = tf.get(t, 0.0) + 1.0
            for t in toks(title or ""):
                tf[t] = tf.get(t, 0.0) + 2.0
            for t in toks(section or ""):
                tf[t] = tf.get(t, 0.0) + 1.2
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
        """Save.

        Parameters
        ----------
        path : str
            TODO.

        Returns
        -------
        None
            TODO.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "N": self.N,
                    "avgdl": self.avgdl,
                    "df": self.df,
                    "docs": self.docs,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> BM25Index:
        """Load.

        Parameters
        ----------
        path : str
            TODO.

        Returns
        -------
        "BM25Index"
            TODO.
        """
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls(d.get("k1", 0.9), d.get("b", 0.4))
        idx.N = d["N"]
        idx.avgdl = d["avgdl"]
        idx.df = d["df"]
        idx.docs = d["docs"]
        return idx

    def _idf(self, term: str) -> float:
        """Idf.

        Parameters
        ----------
        term : str
            TODO.

        Returns
        -------
        float
            TODO.
        """
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0) if self.N > 0 and df > 0 else 0.0

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Search.

        Parameters
        ----------
        query : str
            TODO.
        k : int
            TODO.
        """
        if self.N == 0:
            return []
        terms = toks(query)
        scores = [0.0] * self.N
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in terms:
                tf = d.tf.get(t, 0.0)
                if tf <= 0.0:
                    continue
                idf = self._idf(t)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d.dl / (self.avgdl or 1.0)))
                s += idf * ((tf * (self.k1 + 1.0)) / denom)
            scores[i] = s
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0.0]

    def doc(self, idx: int) -> BM25Doc:
        """Doc.

        Parameters
        ----------
        idx : int
            TODO.

        Returns
        -------
        BM25Doc
            TODO.
        """
        return self.docs[idx]
