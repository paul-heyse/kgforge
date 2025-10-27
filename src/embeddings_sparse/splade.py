"""Module for embeddings_sparse.splade.

NavMap:
- NavMap: Structure describing a module navmap.
- SPLADEv3Encoder: Describe the SPLADE configuration used for neural encoding.
- PureImpactIndex: Approximate SPLADE indexing with TF/IDF-style impact….
- LuceneImpactIndex: Bridge to a Pyserini SPLADE impact index stored on disk.
- get_splade: Construct a SPLADE impact index for the requested backend.
"""

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
    """Describe the SPLADE configuration used for neural encoding."""

    name = "SPLADE-v3-distilbert"

    def __init__(
        self,
        model_id: str = "naver/splade-v3-distilbert",
        device: str = "cuda",
        topk: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        """Record encoder parameters used when exporting SPLADE activations.

        Parameters
        ----------
        model_id : str, optional
            Hugging Face model identifier backing the encoder.
        device : str, optional
            Compute target for inference (``"cuda"`` or ``"cpu"``).
        topk : int, optional
            Maximum number of impact weights retained per document.
        max_seq_len : int, optional
            Maximum sequence length fed to the encoder.
        """
        self.model_id = model_id
        self.device = device
        self.topk = topk
        self.max_seq_len = max_seq_len

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Generate sparse impact vectors for each input text.

        Parameters
        ----------
        texts : list[str]
            Raw text segments to encode.

        Returns
        -------
        list[tuple[list[int], list[float]]]
            Pairs of token identifiers and weights describing the sparse vector.

        Raises
        ------
        NotImplementedError
            Always raised because the skeleton does not ship a neural encoder.
        """
        message = (
            "SPLADE encoding is not implemented in the skeleton. Use the Lucene "
            "impact index variant if available."
        )
        raise NotImplementedError(message)


# [nav:anchor PureImpactIndex]
class PureImpactIndex:
    """Approximate SPLADE indexing with TF/IDF-style impact weighting.

    Keeps the retrieval path runnable without Pyserini or GPU resources by using a simple tokenizer
    and log-scaled weights.
    """

    def __init__(self, index_dir: str) -> None:
        """Prepare an empty impact index rooted at ``index_dir``."""
        self.index_dir = index_dir
        self.df: dict[str, int] = {}
        self.N = 0
        self.postings: dict[str, dict[str, float]] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split ``text`` into lowercase alphanumeric tokens."""
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Construct the impact index from an iterable of documents."""
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
        """Load an existing impact index from disk."""
        with open(os.path.join(self.index_dir, "impact.pkl"), "rb") as handle:
            data = pickle.load(handle)
        self.df = data["df"]
        self.N = data["N"]
        self.postings = data["postings"]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Score documents with impact weights and return the top ``k`` hits."""
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
    """Bridge to a Pyserini SPLADE impact index stored on disk."""

    def __init__(self, index_dir: str) -> None:
        """Store the index location and delay Lucene imports until needed."""
        self.index_dir = index_dir
        self._searcher: Any | None = None

    def _ensure(self) -> None:
        """Initialise the Lucene searcher if it has not been created yet."""
        if self._searcher is not None:
            return
        try:
            from pyserini.search.lucene import LuceneImpactSearcher
        except Exception as exc:  # pragma: no cover - defensive for optional dep
            message = "Pyserini not available for SPLADE impact search"
            raise RuntimeError(message) from exc
        self._searcher = LuceneImpactSearcher(self.index_dir)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Execute a SPLADE impact search against the Lucene index."""
        self._ensure()
        if self._searcher is None:
            message = "Lucene impact searcher not initialized"
            raise RuntimeError(message)
        hits = self._searcher.search(query, k=k)  # expects SPLADE-encoded string
        return [(hit.docid, float(hit.score)) for hit in hits]


# [nav:anchor get_splade]
def get_splade(backend: str, index_dir: str) -> PureImpactIndex | LuceneImpactIndex:
    """Construct a SPLADE impact index for the requested backend."""
    if backend == "lucene":
        try:
            return LuceneImpactIndex(index_dir)
        except Exception:  # pragma: no cover - fallback to pure-python path
            pass
    return PureImpactIndex(index_dir)
