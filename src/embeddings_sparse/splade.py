"""Module for embeddings_sparse.splade.

NavMap:
- SPLADEv3Encoder: Spladev3encoder.
- PureImpactIndex: Toy 'impact' index that approximates SPLADE with IDF/log….
- LuceneImpactIndex: Pyserini SPLADE impact index wrapper.
- get_splade: Get splade.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class SPLADEv3Encoder:
    """Spladev3encoder."""

    name = "SPLADE-v3-distilbert"

    def __init__(
        self,
        model_id: str = "naver/splade-v3-distilbert",
        device: str = "cuda",
        topk: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        """Init.

        Parameters
        ----------
        model_id : str
            TODO.
        device : str
            TODO.
        topk : int
            TODO.
        max_seq_len : int
            TODO.
        """
        self.model_id = model_id
        self.device = device
        self.topk = topk
        self.max_seq_len = max_seq_len

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Encode.

        Parameters
        ----------
        texts : List[str]
            TODO.

        Returns
        -------
        List[Tuple[list[int], list[float]]]
            TODO.
        """
        # Placeholder in this skeleton. Real impl would run the HF model and build
        # top-k (token_id, weight) pairs from the encoder output.
        message = (
            "SPLADE encoding is not implemented in the skeleton. Use the Lucene "
            "impact index variant if available."
        )
        raise NotImplementedError(message)


class PureImpactIndex:
    """Toy 'impact' index that approximates SPLADE with IDF/log weighting.

    - Keeps the pipeline runnable without GPUs or Pyserini.
    - Substitutes a simple tokenizer plus weighting scheme for the neural encoder.
    """

    def __init__(self, index_dir: str) -> None:
        """Init.

        Parameters
        ----------
        index_dir : str
            TODO.
        """
        self.index_dir = index_dir
        self.df: dict[str, int] = {}
        self.N = 0
        self.postings: dict[str, dict[str, float]] = {}

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
        postings: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        doc_count = 0
        for doc_id, fields in docs_iterable:
            text = " ".join(
                [fields.get("title", ""), fields.get("section", ""), fields.get("body", "")]
            )
            toks = self._tokenize(text)
            doc_count += 1
            c = Counter(toks)
            for tok, tf in c.items():
                df[tok] += 1
                postings[tok][doc_id] = math.log1p(tf)
        # compute idf and impact weights
        self.N = doc_count
        self.df = dict(df)
        self.postings = {
            tok: {
                doc: w * math.log((doc_count - df[tok] + 0.5) / (df[tok] + 0.5) + 1.0)
                for doc, w in docs.items()
            }
            for tok, docs in postings.items()
        }
        with open(os.path.join(self.index_dir, "impact.pkl"), "wb") as f:
            pickle.dump(
                {"df": self.df, "N": self.N, "postings": self.postings},
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
        with open(os.path.join(self.index_dir, "impact.pkl"), "rb") as f:
            data = pickle.load(f)
        self.df = data["df"]
        self.N = data["N"]
        self.postings = data["postings"]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search.

        Parameters
        ----------
        query : str
            TODO.
        k : int
            TODO.

        Returns
        -------
        List[Tuple[str, float]]
            TODO.
        """
        toks = self._tokenize(query)
        scores: dict[str, float] = defaultdict(float)
        for t in toks:
            posts = self.postings.get(t)
            if not posts:
                continue
            for doc, w in posts.items():
                scores[doc] += w
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


class LuceneImpactIndex:
    """Pyserini SPLADE impact index wrapper.

    Requires Pyserini build step that writes an impact index on disk.
    """

    def __init__(self, index_dir: str) -> None:
        """Init.

        Parameters
        ----------
        index_dir : str
            TODO.
        """
        self.index_dir = index_dir
        self._searcher: Any | None = None

    def _ensure(self) -> None:
        """Ensure."""
        if self._searcher is not None:
            return
        try:
            from pyserini.search.lucene import LuceneImpactSearcher
        except Exception as exc:
            message = "Pyserini not available for SPLADE impact search"
            raise RuntimeError(message) from exc
        self._searcher = LuceneImpactSearcher(self.index_dir)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search.

        Parameters
        ----------
        query : str
            TODO.
        k : int
            TODO.

        Returns
        -------
        List[Tuple[str, float]]
            TODO.
        """
        self._ensure()
        if self._searcher is None:
            message = "Lucene impact searcher not initialized"
            raise RuntimeError(message)
        hits = self._searcher.search(query, k=k)  # expects query to be SPLADE-encoded string
        return [(h.docid, float(h.score)) for h in hits]


def get_splade(backend: str, index_dir: str) -> PureImpactIndex | LuceneImpactIndex:
    """Get splade.

    Parameters
    ----------
    backend : str
        TODO.
    index_dir : str
        TODO.
    """
    if backend == "lucene":
        try:
            return LuceneImpactIndex(index_dir)
        except Exception:
            pass
    return PureImpactIndex(index_dir)
