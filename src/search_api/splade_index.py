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
    """Describe tok.

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
    return [token.lower() for token in TOKEN.findall(text or "")]


# [nav:anchor SpladeDoc]
@dataclass
class SpladeDoc:
    """Describe SpladeDoc.

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
    section : str
        Describe ``section``.
    text : str
        Describe ``text``.
    """

    chunk_id: str
    doc_id: str
    section: str
    text: str


# [nav:anchor SpladeIndex]
class SpladeIndex:
    """Describe SpladeIndex.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    db_path : str
        Describe ``db_path``.
    chunks_dataset_root : str | None, optional
        Describe ``chunks_dataset_root``.
        Defaults to ``None``.
    sparse_root : str | None, optional
        Describe ``sparse_root``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        db_path: str,
        chunks_dataset_root: str | None = None,
        sparse_root: str | None = None,
    ) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        db_path : str
            Describe ``db_path``.
        chunks_dataset_root : str | None, optional
            Describe ``chunks_dataset_root``.
            Defaults to ``None``.
        sparse_root : str | None, optional
            Describe ``sparse_root``.
            Defaults to ``None``.
        """
        _ = sparse_root  # retained for interface compatibility
        self.db_path = db_path
        self.docs: list[SpladeDoc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self._load(chunks_dataset_root)

    def _load(self, chunks_root: str | None) -> None:
        """Describe  load.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        chunks_root : str | None
            Describe ``chunks_root``.
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
        """Describe doc.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index : int
            Describe ``index``.


        Returns
        -------
        SpladeDoc
            Describe return value.
        """
        return self.docs[index]
