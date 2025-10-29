"""Overview of fixture index.

This module bundles fixture index logic for the kgfoundry stack. It
groups related helpers so downstream packages can import a single
cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
    "synopsis": "In-memory fixture index used for tests and tutorials",
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


# [nav:anchor tokenize]
def tokenize(text: str) -> list[str]:
    """Describe tokenize.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
text : str
    Describe ``text``.

Returns
-------
list[str]
    Describe return value.
"""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


# [nav:anchor FixtureDoc]
@dataclass
class FixtureDoc:
    """Describe FixtureDoc.

<!-- auto:docstring-builder v1 -->

how instances collaborate with the surrounding package. Highlight
how the class supports nearby modules to guide readers through the
codebase.

Parameters
----------
chunk_id : str
    Describe ``chunk_id``.
doc_id : str
    Describe ``doc_id``.
title : str
    Describe ``title``.
section : str
    Describe ``section``.
text : str
    Describe ``text``.
"""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


# [nav:anchor FixtureIndex]
class FixtureIndex:
    """Describe FixtureIndex.

<!-- auto:docstring-builder v1 -->

how instances collaborate with the surrounding package. Highlight
how the class supports nearby modules to guide readers through the
codebase.

Parameters
----------
root : str, optional
    Describe ``root``.
    Defaults to ``'/data'``.
db_path : str, optional
    Describe ``db_path``.
    Defaults to ``'/data/catalog/catalog.duckdb'``.
"""

    def __init__(self, root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb") -> None:
        """Describe   init  .

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
root : str, optional
    Describe ``root``.
    Defaults to ``'/data'``.
db_path : str, optional
    Describe ``db_path``.
    Defaults to ``'/data/catalog/catalog.duckdb'``.
"""
        self.root = Path(root)
        self.db_path = db_path
        self.docs: list[FixtureDoc] = []
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self._load_from_duckdb()

    def _load_from_duckdb(self) -> None:
        """Describe  load from duckdb.

<!-- auto:docstring-builder v1 -->

Python's object protocol for this class. Use it to integrate
with built-in operators, protocols, or runtime behaviours that
expect instances to participate in the language's data model.
"""
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
        """Describe  build lex.

<!-- auto:docstring-builder v1 -->

Python's object protocol for this class. Use it to integrate
with built-in operators, protocols, or runtime behaviours that
expect instances to participate in the language's data model.
"""
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
        """Describe search.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
query : str
    Describe ``query``.
k : int, optional
    Describe ``k``.
    Defaults to ``10``.

Returns
-------
list[tuple[int, float]]
    Describe return value.
"""
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
        """Describe doc.

<!-- auto:docstring-builder v1 -->

Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

Parameters
----------
index : int
    Describe ``index``.

Returns
-------
FixtureDoc
    Describe return value.
"""
        return self.docs[index]
