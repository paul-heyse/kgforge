"""Module for embeddings_sparse.bm25.

NavMap:
- BM25Doc: Bm25doc.
- PurePythonBM25: Simple offline BM25 builder & searcher (Okapi BM25).
- LuceneBM25: Pyserini-backed Lucene BM25 adapter.
- get_bm25: Get bm25.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class BM25Doc:
    """Bm25doc."""

    doc_id: str
    length: int
    fields: dict[str, str]


class PurePythonBM25:
    """Simple offline BM25 builder & searcher (Okapi BM25).

    Persisted as a pickle.
    Fields: title, section, body with configurable boosts.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        field_boosts: dict[str, float] | None = None,
    ) -> None:
        """Init.

        Parameters
        ----------
        index_dir : str
            TODO.
        k1 : float
            TODO.
        b : float
            TODO.
        field_boosts : Optional[Dict[str, float]]
            TODO.
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
        return [t.lower() for t in TOKEN_RE.findall(text)]

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Build.

        Parameters
        ----------
        docs_iterable : Iterable[Tuple[str, Dict]]
            TODO.

        Returns
        -------
        None
            TODO.
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
        """Load.

        Returns
        -------
        None
            TODO.
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
        n_t = self.df.get(term, 0)
        if n_t == 0:
            return 0.0
        # BM25 idf variant
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def search(
        self, query: str, k: int, fields: Mapping[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Search.

        Parameters
        ----------
        query : str
            TODO.
        k : int
            TODO.
        fields : Dict | None
            TODO.

        Returns
        -------
        List[Tuple[str, float]]
            TODO.
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


class LuceneBM25:
    """Pyserini-backed Lucene BM25 adapter.

    Lazily imported.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        field_boosts: dict[str, float] | None = None,
    ) -> None:
        """Init.

        Parameters
        ----------
        index_dir : str
            TODO.
        k1 : float
            TODO.
        b : float
            TODO.
        field_boosts : Optional[Dict[str,float]]
            TODO.
        """
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.field_boosts = field_boosts or {"title": 2.0, "section": 1.2, "body": 1.0}
        self._searcher: Any | None = None

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Build.

        Parameters
        ----------
        docs_iterable : Iterable[Tuple[str, Dict]]
            TODO.

        Returns
        -------
        None
            TODO.
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
        """Ensure searcher."""
        if self._searcher is not None:
            return
        from pyserini.search.lucene import LuceneSearcher

        self._searcher = LuceneSearcher(self.index_dir)
        self._searcher.set_bm25(self.k1, self.b)

    def search(
        self, query: str, k: int, fields: dict[str, str] | None = None
    ) -> list[tuple[str, float]]:
        """Search.

        Parameters
        ----------
        query : str
            TODO.
        k : int
            TODO.
        fields : Dict | None
            TODO.

        Returns
        -------
        List[Tuple[str, float]]
            TODO.
        """
        self._ensure_searcher()
        if self._searcher is None:
            message = "Lucene searcher not initialized"
            raise RuntimeError(message)
        hits = self._searcher.search(query, k=k)
        return [(h.docid, float(h.score)) for h in hits]


def get_bm25(
    backend: str,
    index_dir: str,
    *,
    k1: float = 0.9,
    b: float = 0.4,
    field_boosts: dict[str, float] | None = None,
) -> PurePythonBM25 | LuceneBM25:
    """Get bm25.

    Parameters
    ----------
    backend : str
        TODO.
    index_dir : str
        TODO.
    """
    if backend == "lucene":
        try:
            return LuceneBM25(index_dir, k1=k1, b=b, field_boosts=field_boosts)
        except Exception:
            # allow fallback creation
            pass
    return PurePythonBM25(index_dir, k1=k1, b=b, field_boosts=field_boosts)
