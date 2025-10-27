"""Lightweight lexical index used by the fixture search API.

NavMap:
- tokenize: Tokenise text for the fixture BM25-lite scorer.
- FixtureDoc: Minimal record describing a chunked document.
- FixtureIndex: Build and query an in-memory lexical index sourced from DuckDB.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import duckdb

from kgfoundry_common.navmap_types import NavMap

__all__ = ["FixtureDoc", "FixtureIndex", "tokenize"]

__navmap__: Final[NavMap] = {
    "title": "search_api.fixture_index",
    "synopsis": "Tiny lexical index backed by DuckDB parquet fixtures.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["tokenize", "FixtureDoc", "FixtureIndex"],
        },
    ],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# [nav:anchor tokenize]
def tokenize(text: str) -> list[str]:
    """Tokenise ``text`` into lowercase alphanumeric tokens."""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


# [nav:anchor FixtureDoc]
@dataclass
class FixtureDoc:
    """Chunk-level document representation used by the lexical index."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


# [nav:anchor FixtureIndex]
class FixtureIndex:
    """Materialise a lexical index from DuckDB parquet exports."""

    def __init__(self, root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb") -> None:
        """Initialise the index and eagerly load chunk metadata."""
        self.root = Path(root)
        self.db_path = db_path
        self.docs: list[FixtureDoc] = []
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self._load_from_duckdb()

    def _load_from_duckdb(self) -> None:
        """Read the latest chunk dataset and populate the document list."""
        if not Path(self.db_path).exists():
            return
        con = duckdb.connect(self.db_path)
        try:
            dataset = con.execute(
                """
              SELECT parquet_root FROM datasets
              WHERE kind='chunks'
              ORDER BY created_at DESC
              LIMIT 1
            """
            ).fetchone()
            if not dataset:
                return
            root = dataset[0]
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
        """Build term-frequency and document-frequency statistics."""
        self.tf.clear()
        self.df.clear()
        for doc in self.docs:
            tokens = tokenize(doc.text)
            tf_counts: dict[str, int] = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1
            self.tf.append(tf_counts)
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1
        self.N = len(self.docs)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Score documents for ``query`` using a simple TF-IDF variant."""
        if getattr(self, "N", 0) == 0:
            return []
        qtoks = tokenize(query)
        if not qtoks:
            return []
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            score = 0.0
            for token in qtoks:
                if token not in self.df:
                    continue
                idf = math.log((self.N + 1) / (self.df[token] + 0.5) + 1.0)
                score += idf * tf.get(token, 0)
            scores[i] = score
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:k] if score > 0.0]

    def doc(self, index: int) -> FixtureDoc:
        """Return the :class:`FixtureDoc` at ``index``."""
        return self.docs[index]
