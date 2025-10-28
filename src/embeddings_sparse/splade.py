"""Splade utilities."""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["LuceneImpactIndex", "PureImpactIndex", "SPLADEv3Encoder", "get_splade"]

__navmap__: Final[NavMap] = {
    "title": "embeddings_sparse.splade",
    "synopsis": "SPLADE sparse embedding helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": [
                "SPLADEv3Encoder",
                "PureImpactIndex",
                "LuceneImpactIndex",
                "get_splade",
            ],
        },
    ],
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# [nav:anchor SPLADEv3Encoder]
class SPLADEv3Encoder:
    """Describe SPLADEv3Encoder."""

    name = "SPLADE-v3-distilbert"

    def __init__(
        self,
        model_id: str = "naver/splade-v3-distilbert",
        device: str = "cuda",
        topk: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        model_id : str | None
            Description for ``model_id``.
        device : str | None
            Description for ``device``.
        topk : int | None
            Description for ``topk``.
        max_seq_len : int | None
            Description for ``max_seq_len``.
        """
        
        
        
        
        
        
        
        
        
        self.model_id = model_id
        self.device = device
        self.topk = topk
        self.max_seq_len = max_seq_len

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Compute encode.

        Carry out the encode operation.

        Parameters
        ----------
        texts : List[str]
            Description for ``texts``.

        Returns
        -------
        List[Tuple[List[int], List[float]]]
            Description of return value.

        Raises
        ------
        NotImplementedError
            Raised when validation fails.
        """
        
        
        
        
        
        
        
        
        
        message = (
            "SPLADE encoding is not implemented in the skeleton. Use the Lucene "
            "impact index variant if available."
        )
        raise NotImplementedError(message)


# [nav:anchor PureImpactIndex]
class PureImpactIndex:
    """Describe PureImpactIndex."""

    def __init__(self, index_dir: str) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        index_dir : str
            Description for ``index_dir``.
        """
        
        
        
        
        
        
        
        
        
        self.index_dir = index_dir
        self.df: dict[str, int] = {}
        self.N = 0
        self.postings: dict[str, dict[str, float]] = {}

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
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Compute build.

        Carry out the build operation.

        Parameters
        ----------
        docs_iterable : Iterable[Tuple[str, dict[str, str]]]
            Description for ``docs_iterable``.
        """
        
        
        
        
        
        
        
        
        
        os.makedirs(self.index_dir, exist_ok=True)
        df: dict[str, int] = defaultdict(int)
        postings: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        doc_count = 0
        for doc_id, fields in docs_iterable:
            text = " ".join(
                [fields.get("title", ""), fields.get("section", ""), fields.get("body", "")]
            )
            tokens = self._tokenize(text)
            doc_count += 1
            counts = Counter(tokens)
            for token, term_freq in counts.items():
                df[token] += 1
                postings[token][doc_id] = math.log1p(term_freq)
        self.N = doc_count
        self.df = dict(df)
        self.postings = {
            token: {
                doc: weight * math.log((doc_count - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
                for doc, weight in docs.items()
            }
            for token, docs in postings.items()
        }
        with open(os.path.join(self.index_dir, "impact.pkl"), "wb") as handle:
            pickle.dump(
                {"df": self.df, "N": self.N, "postings": self.postings},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self) -> None:
        """Compute load.

        Carry out the load operation.
        """
        
        
        
        
        
        
        
        
        
        with open(os.path.join(self.index_dir, "impact.pkl"), "rb") as handle:
            data = pickle.load(handle)
        self.df = data["df"]
        self.N = data["N"]
        self.postings = data["postings"]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation.

        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int
            Description for ``k``.

        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.
        """
        
        
        
        
        
        
        
        
        
        tokens = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        for token in tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            for doc_id, weight in postings.items():
                scores[doc_id] += weight
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


# [nav:anchor LuceneImpactIndex]
class LuceneImpactIndex:
    """Describe LuceneImpactIndex."""

    def __init__(self, index_dir: str) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        index_dir : str
            Description for ``index_dir``.
        """
        
        
        
        
        
        
        
        
        
        self.index_dir = index_dir
        self._searcher: Any | None = None

    def _ensure(self) -> None:
        """Compute ensure.

        Carry out the ensure operation.

        Raises
        ------
        RuntimeError
            Raised when validation fails.
        """
        if self._searcher is not None:
            return
        try:
            from pyserini.search.lucene import LuceneImpactSearcher
        except Exception as exc:  # pragma: no cover - defensive for optional dep
            message = "Pyserini not available for SPLADE impact search"
            raise RuntimeError(message) from exc
        self._searcher = LuceneImpactSearcher(self.index_dir)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation.

        Parameters
        ----------
        query : str
            Description for ``query``.
        k : int
            Description for ``k``.

        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.

        Raises
        ------
        RuntimeError
            Raised when validation fails.
        """
        
        
        
        
        
        
        
        
        
        self._ensure()
        if self._searcher is None:
            message = "Lucene impact searcher not initialized"
            raise RuntimeError(message)
        hits = self._searcher.search(query, k=k)  # expects SPLADE-encoded string
        return [(hit.docid, float(hit.score)) for hit in hits]


# [nav:anchor get_splade]
def get_splade(backend: str, index_dir: str) -> PureImpactIndex | LuceneImpactIndex:
    """Compute get splade.

    Carry out the get splade operation.

    Parameters
    ----------
    backend : str
        Description for ``backend``.
    index_dir : str
        Description for ``index_dir``.

    Returns
    -------
    PureImpactIndex | LuceneImpactIndex
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    if backend == "lucene":
        try:
            return LuceneImpactIndex(index_dir)
        except Exception:  # pragma: no cover - fallback to pure-python path
            pass
    return PureImpactIndex(index_dir)
