"""Overview of splade.

This module bundles splade logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, Protocol, cast

from kgfoundry_common.config import load_config
from kgfoundry_common.errors import DeserializationError
from kgfoundry_common.safe_pickle_v2 import (
    SignedPickleWrapper,
    UnsafeSerializationError,
    load_unsigned_legacy,
)
from kgfoundry_common.serialization import deserialize_json, serialize_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from re import Pattern
    from types import ModuleType

    from kgfoundry_common.navmap_types import NavMap
    from kgfoundry_common.problem_details import JsonValue

TOKEN_RE: Pattern[str] = re.compile(r"[A-Za-z0-9_]+")


def _default_float_dict() -> defaultdict[str, float]:
    """Return a defaultdict that produces ``float`` zeros for missing keys.

    Returns
    -------
    defaultdict[str, float]
        Defaultdict with float factory.
    """
    return defaultdict(float)


def _score_value(item: tuple[str, float]) -> float:
    """Extract the score component from a Lucene impact result tuple.

    Parameters
    ----------
    item : tuple[str, float]
        Impact result tuple.

    Returns
    -------
    float
        Score value.
    """
    return item[1]


def _decode_signing_key() -> bytes | None:
    """Decode the configured signing key, returning ``None`` if unavailable.

    Returns
    -------
    bytes | None
        Decoded signing key, or None if unavailable.
    """
    try:
        settings = load_config()
    except ValueError as exc:
        logger.warning("Configuration invalid; proceeding without signing key", exc_info=exc)
        return None

    encoded_key = settings.signing_key
    if encoded_key is None:
        return None

    try:
        return base64.b64decode(encoded_key)
    except binascii.Error as exc:
        logger.warning(
            "Signing key is not valid base64; ignoring secure pickle signature", exc_info=exc
        )
        return None


def _load_unsigned_payload(handle: BinaryIO, legacy_path: Path) -> dict[str, JsonValue]:
    """Load legacy pickle payload with allow-list enforcement.

    Parameters
    ----------
    handle : BinaryIO
        Binary file handle to read from.
    legacy_path : Path
        Path to the legacy pickle file.

    Returns
    -------
    dict[str, JsonValue]
        Deserialized payload dictionary.

    Raises
    ------
    DeserializationError
        If deserialization fails or payload is invalid.
    """
    try:
        payload_obj = load_unsigned_legacy(handle)
    except UnsafeSerializationError as exc:
        msg = f"Legacy pickle at {legacy_path} failed safety validation"
        raise DeserializationError(msg) from exc

    if not isinstance(payload_obj, dict):
        msg = f"Invalid legacy pickle payload: expected dict, got {type(payload_obj)}"
        raise DeserializationError(msg)

    return cast("dict[str, JsonValue]", payload_obj)


def _load_legacy_metadata(legacy_path: Path) -> dict[str, JsonValue]:
    """Load SPLADE legacy pickle metadata using signed or unsigned safe loader.

    Parameters
    ----------
    legacy_path : Path
        Path to the legacy pickle file.

    Returns
    -------
    dict[str, JsonValue]
        Deserialized metadata dictionary.

    Raises
    ------
    DeserializationError
        If deserialization fails or payload is invalid.
    OSError
        If the file cannot be read.
    """
    signing_key = _decode_signing_key()
    try:
        with legacy_path.open("rb") as handle:
            if signing_key:
                wrapper = SignedPickleWrapper(signing_key)
                try:
                    payload_obj = wrapper.load(handle)
                except UnsafeSerializationError:
                    logger.warning(
                        "Signed pickle validation failed for legacy SPLADE index; falling back to unsigned loader",
                        extra={"legacy_path": str(legacy_path)},
                    )
                    handle.seek(0)
                    payload = _load_unsigned_payload(handle, legacy_path)
                else:
                    if not isinstance(payload_obj, dict):
                        msg = (
                            f"Invalid signed legacy payload: expected dict, got {type(payload_obj)}"
                        )
                        raise DeserializationError(msg)
                    payload = cast("dict[str, JsonValue]", payload_obj)
            else:
                logger.warning(
                    "Missing signing key; using unsigned legacy pickle loader",
                    extra={"legacy_path": str(legacy_path)},
                )
                payload = _load_unsigned_payload(handle, legacy_path)
    except OSError as exc:
        msg = f"Failed to read legacy SPLADE index at {legacy_path}: {exc}"
        raise DeserializationError(msg) from exc

    return payload


class ImpactHitProtocol(Protocol):
    """Protocol describing the minimal SPLADE impact hit payload."""

    docid: str
    score: float


class LuceneImpactSearcherProtocol(Protocol):
    """Protocol for Lucene-backed SPLADE searchers."""

    def __init__(self, index_dir: str, *, query_encoder: str | None = None) -> None:
        """Initialise the searcher with an index directory and optional encoder."""
        ...

    def search(self, query: str, k: int) -> Sequence[ImpactHitProtocol]:
        """Return the top ``k`` SPLADE impact hits for ``query``."""
        ...


class LuceneImpactSearcherFactory(Protocol):
    """Factory protocol for constructing Lucene SPLADE searchers."""

    def __call__(
        self, index_dir: str, *, query_encoder: str | None = None
    ) -> LuceneImpactSearcherProtocol:
        """Build a searcher for ``index_dir`` using an optional query encoder."""
        ...


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

        Raises
        ------
        NotImplementedError
            This function is not implemented in the skeleton.

        Notes
        -----
        Use the Lucene impact index variant if available.
        """
        message = (
            "SPLADE encoding is not implemented in the skeleton. Use the Lucene "
            "impact index variant if available. "
            f"Requested device={self.device!r} with {len(texts)} texts."
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
        matches = cast("list[str]", TOKEN_RE.findall(text))
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
            except DeserializationError:
                logger.warning(
                    "Failed to load JSON index, trying legacy pickle",
                    extra={"metadata_path": str(metadata_path), "legacy_path": str(legacy_path)},
                )
                if legacy_path.exists():
                    data_dict = _load_legacy_metadata(legacy_path)
                else:
                    raise
            else:
                if not isinstance(data_raw, dict):
                    msg = f"Invalid index data format: expected dict, got {type(data_raw)}"
                    raise DeserializationError(msg)
                data_dict = cast("dict[str, JsonValue]", data_raw)
        elif legacy_path.exists():
            data_dict = _load_legacy_metadata(legacy_path)
            logger.warning("Loaded legacy pickle index. Consider migrating to JSON format.")
        else:
            msg = f"Index metadata not found at {metadata_path} or {legacy_path}"
            raise FileNotFoundError(msg)

        # Extract values with type narrowing
        df_val: JsonValue = data_dict.get("df", {})
        n_val: JsonValue = data_dict.get("N", 0)
        postings_val: JsonValue = data_dict.get("postings", {})

        # Type narrowing and conversion
        self.df = cast("dict[str, int]", df_val) if isinstance(df_val, dict) else {}
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
            lucene_search_module: ModuleType = importlib.import_module("pyserini.search.lucene")
            lucene_impact_searcher_cls = cast(
                "LuceneImpactSearcherFactory",
                lucene_search_module.LuceneImpactSearcher,
            )
        except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
            message = "Pyserini not available for SPLADE impact search"
            logger.exception("Failed to import LuceneImpactSearcher")
            raise RuntimeError(message) from exc
        searcher = lucene_impact_searcher_cls(self.index_dir, query_encoder=self.query_encoder)
        self._searcher = searcher

    def ensure_available(self) -> None:
        """Ensure the Lucene searcher is initialized and ready for queries."""
        self._ensure()

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
        return [(str(hit.docid), float(hit.score)) for hit in hits]


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
        lucene_index = LuceneImpactIndex(index_dir=index_dir, query_encoder=query_encoder)
        try:
            lucene_index.ensure_available()
        except RuntimeError as exc:
            logger.warning(
                "Lucene backend unavailable, falling back to PureImpactIndex",
                extra={"index_dir": index_dir, "query_encoder": query_encoder},
                exc_info=exc,
            )
        else:
            return lucene_index
    return PureImpactIndex(index_dir)
