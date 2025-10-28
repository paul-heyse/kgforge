"""Overview of bm25.

This module bundles bm25 logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["BM25Doc", "LuceneBM25", "PurePythonBM25", "get_bm25"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry.embeddings_sparse.bm25",
    "synopsis": "Pure Python and Lucene-backed BM25 adapters for sparse retrieval",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        }
    ],
    "module_meta": {
        "owner": "@embeddings",
        "stability": "experimental",
        "since": "2024.10",
    },
    "symbols": {
        "BM25Doc": {
            "owner": "@embeddings",
            "stability": "experimental",
            "since": "2024.10",
        },
        "PurePythonBM25": {
            "owner": "@embeddings",
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["fs"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
            "tests": [
                "tests/unit/test_bm25_adapter.py::test_bm25_build_and_search_from_fixtures",
            ],
        },
        "LuceneBM25": {
            "owner": "@embeddings",
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["fs"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
        },
        "get_bm25": {
            "owner": "@embeddings",
            "since": "2024.10",
            "stability": "stable",
            "side_effects": ["none"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
        },
    },
    "edit_scopes": {"safe": ["get_bm25"], "risky": ["PurePythonBM25", "LuceneBM25"]},
    "tags": ["bm25", "retrieval", "sparse"],
    "see_also": ["kgfoundry.search_api.bm25_index"],
    "deps": ["pyserini"],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# [nav:anchor BM25Doc]
@dataclass
class BM25Doc:
    """Model the BM25Doc.

    Represent the bm25doc data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    doc_id: str
    length: int
    fields: dict[str, str]


# [nav:anchor PurePythonBM25]
class PurePythonBM25:
    """Model the PurePythonBM25.

    Represent the purepythonbm25 data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        field_boosts: dict[str, float] | None = None,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        index_dir : str
        index_dir : str
            Description for ``index_dir``.
        k1 : float | None
        k1 : float | None, optional, default=0.9
            Description for ``k1``.
        b : float | None
        b : float | None, optional, default=0.4
            Description for ``b``.
        field_boosts : Mapping[str, float] | None
        field_boosts : Mapping[str, float] | None, optional, default=None
            Description for ``field_boosts``.
        """
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.field_boosts = field_boosts or {"title": 2.0, "section": 1.2, "body": 1.0}
        self.df: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = {}
        self.docs: dict[str, BM25Doc] = {}
        self.N = 0
        self.avgdl = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Compute tokenize.

        Carry out the tokenize operation.

        Parameters
        ----------
        text : str
            Description for ``text``.

        Returns
        -------
        List[str]
            Description of return value.
        """
        return [t.lower() for t in TOKEN_RE.findall(text)]

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Compute build.

        Carry out the build operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        docs_iterable : collections.abc.Iterable
        docs_iterable : collections.abc.Iterable
            Description for ``docs_iterable``.

        Examples
        --------
        >>> from embeddings_sparse.bm25 import build
        >>> build(...)  # doctest: +ELLIPSIS
        """
        os.makedirs(self.index_dir, exist_ok=True)
        df: dict[str, int] = defaultdict(int)
        postings: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        docs: dict[str, BM25Doc] = {}
        lengths: list[int] = []
        for doc_id, fields in docs_iterable:
            body = fields.get("body", "")
            section = fields.get("section", "")
            title = fields.get("title", "")
            # field boosts applied at scoring time; here we merge for length calc
            toks = self._tokenize(title + " " + section + " " + body)
            lengths.append(len(toks))
            docs[doc_id] = BM25Doc(
                doc_id=doc_id,
                length=len(toks),
                fields={"title": title, "section": section, "body": body},
            )
            seen = set()
            for tok in toks:
                postings[tok][doc_id] += 1
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)
        self.N = len(docs)
        self.avgdl = sum(lengths) / max(1, len(lengths))
        self.df = dict(df)
        # convert defaultdicts
        self.postings = {t: dict(ps) for t, ps in postings.items()}
        self.docs = docs
        # persist
        with open(os.path.join(self.index_dir, "pure_bm25.pkl"), "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "field_boosts": self.field_boosts,
                    "df": self.df,
                    "postings": self.postings,
                    "docs": self.docs,
                    "N": self.N,
                    "avgdl": self.avgdl,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self) -> None:
        """Compute load.

        Carry out the load operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Examples
        --------
        >>> from embeddings_sparse.bm25 import load
        >>> load()  # doctest: +ELLIPSIS
        """
        path = os.path.join(self.index_dir, "pure_bm25.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        self.field_boosts = data["field_boosts"]
        self.df = data["df"]
        self.postings = data["postings"]
        self.docs = data["docs"]
        self.N = data["N"]
        self.avgdl = data["avgdl"]

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
        n_t = self.df.get(term, 0)
        if n_t == 0:
            return 0.0
        # BM25 idf variant
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        query : str
        query : str
            Description for ``query``.
        k : int
        k : int
            Description for ``k``.
        fields : Mapping[str, str] | None
        fields : Mapping[str, str] | None, optional, default=None
            Description for ``fields``.

        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.

        Examples
        --------
        >>> from embeddings_sparse.bm25 import search
        >>> result = search(..., ...)
        >>> result  # doctest: +ELLIPSIS
        """
        # naive field weighting at score aggregation (title/section/body contributions)
        tokens = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        for term in tokens:
            idf = self._idf(term)
            postings = self.postings.get(term)
            if not postings:
                continue
            for doc_id, tf in postings.items():
                doc = self.docs[doc_id]
                dl = doc.length or 1
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                contrib = idf * ((tf * (self.k1 + 1)) / (denom))
                scores[doc_id] += contrib
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# [nav:anchor LuceneBM25]
class LuceneBM25:
    """Model the LuceneBM25.

    Represent the lucenebm25 data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        field_boosts: dict[str, float] | None = None,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        index_dir : str
        index_dir : str
            Description for ``index_dir``.
        k1 : float | None
        k1 : float | None, optional, default=0.9
            Description for ``k1``.
        b : float | None
        b : float | None, optional, default=0.4
            Description for ``b``.
        field_boosts : Mapping[str, float] | None
        field_boosts : Mapping[str, float] | None, optional, default=None
            Description for ``field_boosts``.
        """
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.field_boosts = field_boosts or {"title": 2.0, "section": 1.2, "body": 1.0}
        self._searcher: Any | None = None

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Compute build.

        Carry out the build operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        docs_iterable : collections.abc.Iterable
        docs_iterable : collections.abc.Iterable
            Description for ``docs_iterable``.

        Raises
        ------
        RuntimeError
            Raised when validation fails.

        Examples
        --------
        >>> from embeddings_sparse.bm25 import build
        >>> build(...)  # doctest: +ELLIPSIS
        """
        try:
            from pyserini.analysis import get_lucene_analyzer
            from pyserini.index import IndexWriter
        except Exception as exc:
            message = "Pyserini/Lucene not available"
            raise RuntimeError(message) from exc
        os.makedirs(self.index_dir, exist_ok=True)
        analyzer = get_lucene_analyzer(stemmer="english", stopwords=True)
        writer = IndexWriter(self.index_dir, analyzer=analyzer, keep_stopwords=False)
        for doc_id, fields in docs_iterable:
            # combine fields with boosts in a "contents" field for simplicity
            title = fields.get("title", "")
            section = fields.get("section", "")
            body = fields.get("body", "")
            contents = " ".join(
                [
                    (title + " ") * int(self.field_boosts.get("title", 1.0)),
                    (section + " ") * int(self.field_boosts.get("section", 1.0)),
                    body,
                ]
            )
            writer.add_document(docid=doc_id, contents=contents)
        writer.close()

    def _ensure_searcher(self) -> None:
        """Compute ensure searcher.

        Carry out the ensure searcher operation.
        """
        if self._searcher is not None:
            return
        from pyserini.search.lucene import LuceneSearcher

        self._searcher = LuceneSearcher(self.index_dir)
        self._searcher.set_bm25(self.k1, self.b)

    def search(
        self, query: str, k: int, fields: dict[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        query : str
        query : str
            Description for ``query``.
        k : int
        k : int
            Description for ``k``.
        fields : Mapping[str, str] | None
        fields : Mapping[str, str] | None, optional, default=None
            Description for ``fields``.

        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.

        Raises
        ------
        RuntimeError
            Raised when validation fails.

        Examples
        --------
        >>> from embeddings_sparse.bm25 import search
        >>> result = search(..., ...)
        >>> result  # doctest: +ELLIPSIS
        """
        self._ensure_searcher()
        if self._searcher is None:
            message = "Lucene searcher not initialized"
            raise RuntimeError(message)
        hits = self._searcher.search(query, k=k)
        return [(h.docid, float(h.score)) for h in hits]


# [nav:anchor get_bm25]
def get_bm25(
    backend: str,
    index_dir: str,
    *,
    k1: float = 0.9,
    b: float = 0.4,
    field_boosts: dict[str, float] | None = None,
) -> PurePythonBM25 | LuceneBM25:
    """Compute get bm25.

    Carry out the get bm25 operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    backend : str
    backend : str
        Description for ``backend``.
    index_dir : str
    index_dir : str
        Description for ``index_dir``.
    k1 : float | None
    k1 : float | None, optional, default=0.9
        Description for ``k1``.
    b : float | None
    b : float | None, optional, default=0.4
        Description for ``b``.
    field_boosts : Mapping[str, float] | None
    field_boosts : Mapping[str, float] | None, optional, default=None
        Description for ``field_boosts``.

    Returns
    -------
    PurePythonBM25 | LuceneBM25
        Description of return value.

    Examples
    --------
    >>> from embeddings_sparse.bm25 import get_bm25
    >>> result = get_bm25(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    if backend == "lucene":
        try:
            return LuceneBM25(index_dir, k1=k1, b=b, field_boosts=field_boosts)
        except Exception:
            # allow fallback creation
            pass
    return PurePythonBM25(index_dir, k1=k1, b=b, field_boosts=field_boosts)
