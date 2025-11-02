"""Overview of splade.

This module bundles splade logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Final, Protocol, cast

if TYPE_CHECKING:
    pass

from kgfoundry_common.errors import DeserializationError
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import JsonValue
from kgfoundry_common.serialization import deserialize_json, serialize_json

TOKEN_RE: Pattern[str] = re.compile(r"[A-Za-z0-9_]+")


def _default_float_dict() -> defaultdict[str, float]:
    return defaultdict(float)


def _score_value(item: tuple[str, float]) -> float:
    return item[1]


class ImpactHitProtocol(Protocol):
    docid: str
    score: float


class LuceneImpactSearcherProtocol(Protocol):
    def search(self, query: str, k: int) -> Sequence[ImpactHitProtocol]: ...


logger = logging.getLogger(__name__)

__all__ = ["LuceneImpactIndex", "PureImpactIndex", "SPLADEv3Encoder", "get_splade"]

__navmap__: Final[NavMap] = {
    "title": "embeddings_sparse.splade",
    "synopsis": "SPLADE sparse embedding helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@embeddings",
        "stability": "experimental",
        "since": "2024.10",
    },
    "symbols": {
        "SPLADEv3Encoder": {
            "owner": "@embeddings",
            "stability": "experimental",
            "since": "2024.10",
        },
        "PureImpactIndex": {
            "owner": "@embeddings",
            "stability": "experimental",
            "since": "2024.10",
            "side_effects": ["fs"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
        },
        "LuceneImpactIndex": {
            "owner": "@embeddings",
            "stability": "experimental",
            "since": "2024.10",
            "side_effects": ["fs"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
        },
        "get_splade": {
            "owner": "@embeddings",
            "stability": "experimental",
            "since": "2024.10",
        },
    },
}


# [nav:anchor SPLADEv3Encoder]
class SPLADEv3Encoder:
    """Describe SPLADEv3Encoder.

    <!-- auto:docstring-builder v1 -->

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.

    Parameters
    ----------
    model_id : str, optional
        Describe ``model_id``.
        Defaults to ``'naver/splade-v3-distilbert'``.
    device : str, optional
        Describe ``device``.
        Defaults to ``'cuda'``.
    topk : int, optional
        Describe ``topk``.
        Defaults to ``256``.
    max_seq_len : int, optional
        Describe ``max_seq_len``.
        Defaults to ``512``.

    Raises
    ------
    NotImplementedError
    Raised when TODO for NotImplementedError.
    """

    name = "SPLADE-v3-distilbert"

    def __init__(
        self,
        model_id: str = "naver/splade-v3-distilbert",
        device: str = "cuda",
        topk: int = 256,
        max_seq_len: int = 512,
    ) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        model_id : str, optional
            Describe ``model_id``.
            Defaults to ``'naver/splade-v3-distilbert'``.
        device : str, optional
            Describe ``device``.
            Defaults to ``'cuda'``.
        topk : int, optional
            Describe ``topk``.
            Defaults to ``256``.
        max_seq_len : int, optional
            Describe ``max_seq_len``.
            Defaults to ``512``.
        """
        self.model_id = model_id
        self.device = device
        self.topk = topk
        self.max_seq_len = max_seq_len

    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Describe encode.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        texts : list[str]
            Describe ``texts``.

        Returns
        -------
        list[tuple[list[int], list[float]]]
            Describe return value.

        Raises
        ------
        NotImplementedError
        Raised when TODO for NotImplementedError.
        """
        message = (
            "SPLADE encoding is not implemented in the skeleton. Use the Lucene "
            "impact index variant if available."
        )
        raise NotImplementedError(message)


# [nav:anchor PureImpactIndex]
class PureImpactIndex:
    """Describe PureImpactIndex.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    index_dir : str
        Describe ``index_dir``.
    """

    def __init__(self, index_dir: str) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index_dir : str
            Describe ``index_dir``.
        """
        self.index_dir = index_dir
        self.df: dict[str, int] = {}
        self.N = 0
        self.postings: dict[str, dict[str, float]] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Describe  tokenize.

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
        matches = cast(list[str], TOKEN_RE.findall(text))
        return [token.lower() for token in matches]

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Describe build.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        docs_iterable : tuple[str, dict[str, str]]
            Describe ``docs_iterable``.
        """
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        df: defaultdict[str, int] = defaultdict(int)
        postings: defaultdict[str, defaultdict[str, float]] = defaultdict(_default_float_dict)
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
        # persist using secure JSON serialization with schema validation
        metadata_path = Path(self.index_dir) / "impact.json"
        schema_path = (
            Path(__file__).parent.parent.parent / "schema" / "models" / "splade_metadata.v1.json"
        )
        payload = {"df": self.df, "N": self.N, "postings": self.postings}
        serialize_json(payload, schema_path, metadata_path)

    def load(self) -> None:
        """Load SPLADE index metadata from disk with schema validation and checksum verification.

        <!-- auto:docstring-builder v1 -->

        Deserializes index metadata from JSON, verifying checksum and validating
        against the schema. Falls back to legacy pickle format for backward compatibility.

        Raises
        ------
        DeserializationError
            If deserialization, schema validation, or checksum verification fails.
        FileNotFoundError
            If metadata or schema file is missing.
        """
        metadata_path = Path(self.index_dir) / "impact.json"
        schema_path = (
            Path(__file__).parent.parent.parent / "schema" / "models" / "splade_metadata.v1.json"
        )
        legacy_path = Path(self.index_dir) / "impact.pkl"

        if metadata_path.exists():
            try:
                data_raw = deserialize_json(metadata_path, schema_path)
            except DeserializationError as exc:
                logger.warning("Failed to load JSON index, trying legacy pickle: %s", exc)
                # Fall back to legacy pickle
                if legacy_path.exists():
                    import pickle

                    with legacy_path.open("rb") as handle:
                        data_raw = pickle.load(handle)  # noqa: S301
                else:
                    raise
        elif legacy_path.exists():
            # Legacy pickle format
            import pickle

            with legacy_path.open("rb") as handle:
                data_raw = pickle.load(handle)  # noqa: S301
            logger.warning("Loaded legacy pickle index. Consider migrating to JSON format.")
        else:
            msg = f"Index metadata not found at {metadata_path} or {legacy_path}"
            raise FileNotFoundError(msg)

        # Narrow data type before indexing - pickle.load returns object
        if not isinstance(data_raw, dict):
            msg = f"Invalid pickle data format: expected dict, got {type(data_raw)}"
            raise DeserializationError(msg)
        data_dict: dict[str, JsonValue] = cast(dict[str, JsonValue], data_raw)

        # Extract values with type narrowing
        df_val: JsonValue = data_dict.get("df", {})
        n_val: JsonValue = data_dict.get("N", 0)
        postings_val: JsonValue = data_dict.get("postings", {})

        # Type narrowing and conversion
        self.df = cast(dict[str, int], df_val) if isinstance(df_val, dict) else {}
        self.N = int(n_val) if isinstance(n_val, (int, float)) else 0
        # postings is dict[str, dict[str, float]] - need to convert nested dict values
        if isinstance(postings_val, dict):
            # Convert nested dict values from JsonValue to dict[str, float]
            postings_converted: dict[str, dict[str, float]] = {}
            for term, term_postings in postings_val.items():
                if isinstance(term_postings, dict):
                    # Convert inner dict values to float
                    term_postings_float: dict[str, float] = {
                        str(doc_id): float(weight) if isinstance(weight, (int, float)) else 0.0
                        for doc_id, weight in term_postings.items()
                    }
                    postings_converted[str(term)] = term_postings_float
            self.postings = postings_converted
        else:
            self.postings = {}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Describe search.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int
            Describe ``k``.

        Returns
        -------
        list[tuple[str, float]]
            Describe return value.
        """
        tokens = self._tokenize(query)
        scores: defaultdict[str, float] = defaultdict(float)
        for token in tokens:
            postings = self.postings.get(token)
            if not postings:
                continue
            for doc_id, weight in postings.items():
                scores[doc_id] += weight
        ranked_scores: list[tuple[str, float]] = [
            (doc_id, score) for doc_id, score in scores.items()
        ]
        ranked_scores.sort(key=_score_value, reverse=True)
        return ranked_scores[:k]


# [nav:anchor LuceneImpactIndex]
class LuceneImpactIndex:
    """Describe LuceneImpactIndex.

    <!-- auto:docstring-builder v1 -->

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.

    Parameters
    ----------
    index_dir : str
        Describe ``index_dir``.
    query_encoder : str, optional
        Describe ``query_encoder``.
        Defaults to ``'naver/splade-v3-distilbert'``.

    Raises
    ------
    RuntimeError
    Raised when TODO for RuntimeError.
    """

    def __init__(self, index_dir: str, query_encoder: str = "naver/splade-v3-distilbert") -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index_dir : str
            Describe ``index_dir``.
        query_encoder : str, optional
            Describe ``query_encoder``.
            Defaults to ``'naver/splade-v3-distilbert'``.
        """
        self.index_dir = index_dir
        self.query_encoder = query_encoder
        self._searcher: LuceneImpactSearcherProtocol | None = None

    def _ensure(self) -> None:
        """Describe  ensure.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
        """
        if self._searcher is not None:
            return
        try:
            from pyserini.search.lucene import LuceneImpactSearcher
        except Exception as exc:  # pragma: no cover - defensive for optional dep
            message = "Pyserini not available for SPLADE impact search"
            logger.exception("Failed to import LuceneImpactSearcher")
            raise RuntimeError(message) from exc
        searcher = LuceneImpactSearcher(self.index_dir, query_encoder=self.query_encoder)
        self._searcher = cast(LuceneImpactSearcherProtocol, searcher)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Describe search.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int
            Describe ``k``.

        Returns
        -------
        list[tuple[str, float]]
            Describe return value.

        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
        """
        self._ensure()
        if self._searcher is None:
            message = "Lucene impact searcher not initialized"
            raise RuntimeError(message)
        hits = self._searcher.search(query, k)
        results: list[tuple[str, float]] = []
        for hit in hits:
            results.append((str(hit.docid), float(hit.score)))
        return results


# [nav:anchor get_splade]
def get_splade(
    backend: str,
    index_dir: str,
    query_encoder: str = "naver/splade-v3-distilbert",
) -> PureImpactIndex | LuceneImpactIndex:
    """Describe get splade.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    backend : str
        Describe ``backend``.
    index_dir : str
        Describe ``index_dir``.
    query_encoder : str, optional
        Describe ``query_encoder``.
        Defaults to ``'naver/splade-v3-distilbert'``.

    Returns
    -------
    PureImpactIndex | LuceneImpactIndex
        Describe return value.
    """
    if backend == "lucene":
        try:
            return LuceneImpactIndex(index_dir=index_dir, query_encoder=query_encoder)
        except Exception as exc:
            logger.warning(
                "Failed to create LuceneImpactIndex, falling back to PureImpactIndex: %s",
                exc,
                exc_info=True,
            )
    return PureImpactIndex(index_dir)
