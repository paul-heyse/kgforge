"""Module for search_api.fixture_index.

NavMap:
- tokenize: Tokenize.
- FixtureDoc: Fixturedoc.
- FixtureIndex: Fixtureindex.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize.

    Parameters
    ----------
    text : str
        TODO.

    Returns
    -------
    List[str]
        TODO.
    """
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class FixtureDoc:
    """Fixturedoc."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


class FixtureIndex:
    """Fixtureindex."""

    def __init__(self, root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb") -> None:
        """Init.

        Parameters
        ----------
        root : str
            TODO.
        db_path : str
            TODO.
        """
        self.root = Path(root)
        self.db_path = db_path
        self.docs: list[FixtureDoc] = []
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self._load_from_duckdb()

    def _load_from_duckdb(self) -> None:
        """Load from duckdb.

        Returns
        -------
        None
            TODO.
        """
        if not Path(self.db_path).exists():
            return
        con = duckdb.connect(self.db_path)
        try:
            ds = con.execute(
                """
              SELECT parquet_root FROM datasets
              WHERE kind='chunks'
              ORDER BY created_at DESC
              LIMIT 1
            """
            ).fetchone()
            if not ds:
                return
            root = ds[0]
            rows = con.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text,
                       coalesce(d.title,'') AS title
                FROM read_parquet('{root}/*/*.parquet', union_by_name=true) AS c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """
            ).fetchall()
        finally:
            con.close()

        for chunk_id, doc_id, section, text, title in rows:
            self.docs.append(
                FixtureDoc(
                    chunk_id=chunk_id,
                    doc_id=doc_id or "urn:doc:fixture",
                    title=title or "Fixture",
                    section=section or "",
                    text=text or "",
                )
            )

        self._build_lex()

    def _build_lex(self) -> None:
        """Build lex.

        Returns
        -------
        None
            TODO.
        """
        self.tf.clear()
        self.df.clear()
        for doc in self.docs:
            toks = tokenize(doc.text)
            tf_counts: dict[str, int] = {}
            for t in toks:
                tf_counts[t] = tf_counts.get(t, 0) + 1
            self.tf.append(tf_counts)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Search.

        Parameters
        ----------
        query : str
            TODO.
        k : int
            TODO.

        Returns
        -------
        List[Tuple[int, float]]
            TODO.
        """
        if getattr(self, "N", 0) == 0:
            return []
        qtoks = tokenize(query)
        if not qtoks:
            return []
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in qtoks:
                if t not in self.df:
                    continue
                idf = math.log((self.N + 1) / (self.df[t] + 0.5) + 1.0)
                s += idf * tf.get(t, 0)
            scores[i] = s
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0.0]

    def doc(self, idx: int) -> FixtureDoc:
        """Doc.

        Parameters
        ----------
        idx : int
            TODO.

        Returns
        -------
        FixtureDoc
            TODO.
        """
        return self.docs[idx]
