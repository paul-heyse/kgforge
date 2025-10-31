"""Hybrid search utilities for the agent catalog."""

from __future__ import annotations

import collections
import importlib
import json
import os
import pickle
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from kgfoundry_common.errors import AgentCatalogSearchError
from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.observability import MetricsProvider, observe_duration
from search_api.types import (
    FaissIndexProtocol,
    FaissModuleProtocol,
    IndexArray,
    VectorArray,
)

logger = get_logger(__name__)


EMBEDDING_MATRIX_RANK = 2
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingModelProtocol(Protocol):
    """Protocol describing the embedding model encode interface.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def encode(self, sentences: Sequence[str], **_: object) -> VectorArray:
        """Return embeddings for the provided sentences.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        sentences : str
            Describe ``sentences``.
        **_ : object
            Describe ``_``.

        Returns
        -------
        tuple[int, ...] | np.float32
            Describe return value.
"""
        ...


LEXICAL_FIELDS = [
    "qname",
    "module",
    "package",
    "summary",
    "docstring",
    "agent_hints.intent_tags",
]


@dataclass(slots=True)
class SearchConfig:
    """Configuration used for hybrid search against the catalog.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    alpha : float
        Describe ``alpha``.
    candidate_pool : int
        Describe ``candidate_pool``.
    lexical_fields : list[str]
        Describe ``lexical_fields``.
"""

    alpha: float
    candidate_pool: int
    lexical_fields: list[str]


@dataclass(slots=True)
class SearchOptions:
    """Optional tuning parameters for hybrid search.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    alpha : float | NoneType, optional
        Describe ``alpha``.
        Defaults to ``None``.
    facets : str | str | NoneType, optional
        Describe ``facets``.
        Defaults to ``None``.
    candidate_pool : int | NoneType, optional
        Describe ``candidate_pool``.
        Defaults to ``None``.
    model_loader : [<class 'str'>] | EmbeddingModelProtocol | NoneType, optional
        Describe ``model_loader``.
        Defaults to ``None``.
    embedding_model : str | NoneType, optional
        Describe ``embedding_model``.
        Defaults to ``None``.
    batch_size : int | NoneType, optional
        Describe ``batch_size``.
        Defaults to ``None``.
"""

    alpha: float | None = None
    facets: Mapping[str, str] | None = None
    candidate_pool: int | None = None
    model_loader: Callable[[str], EmbeddingModelProtocol] | None = None
    embedding_model: str | None = None
    batch_size: int | None = None


@dataclass(slots=True)
class SearchDocument:
    """Intermediate representation used to build or query the semantic index.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    symbol_id : str
        Describe ``symbol_id``.
    package : str
        Describe ``package``.
    module : str
        Describe ``module``.
    qname : str
        Describe ``qname``.
    kind : str
        Describe ``kind``.
    stability : str | NoneType
        Describe ``stability``.
    deprecated : bool
        Describe ``deprecated``.
    summary : str | NoneType
        Describe ``summary``.
    docstring : str | NoneType
        Describe ``docstring``.
    anchor_start : int | NoneType
        Describe ``anchor_start``.
    anchor_end : int | NoneType
        Describe ``anchor_end``.
    text : str
        Describe ``text``.
    tokens : str
        Describe ``tokens``.
    row : int, optional
        Describe ``row``.
        Defaults to ``-1``.
"""

    symbol_id: str
    package: str
    module: str
    qname: str
    kind: str
    stability: str | None
    deprecated: bool
    summary: str | None
    docstring: str | None
    anchor_start: int | None
    anchor_end: int | None
    text: str
    tokens: collections.Counter[str]
    row: int = -1


@dataclass(slots=True)
class SearchResult:
    """Result record returned by hybrid search.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    symbol_id : str
        Describe ``symbol_id``.
    score : float
        Describe ``score``.
    lexical_score : float
        Describe ``lexical_score``.
    vector_score : float
        Describe ``vector_score``.
    package : str
        Describe ``package``.
    module : str
        Describe ``module``.
    qname : str
        Describe ``qname``.
    kind : str
        Describe ``kind``.
    stability : str | NoneType
        Describe ``stability``.
    deprecated : bool
        Describe ``deprecated``.
    summary : str | NoneType
        Describe ``summary``.
    docstring : str | NoneType
        Describe ``docstring``.
    anchor : dict[str, int | NoneType]
        Describe ``anchor``.
"""

    symbol_id: str
    score: float
    lexical_score: float
    vector_score: float
    package: str
    module: str
    qname: str
    kind: str
    stability: str | None
    deprecated: bool
    summary: str | None
    docstring: str | None
    anchor: dict[str, int | None]


@dataclass(slots=True)
class VectorSearchContext:
    """Data bundle required for computing vector scores.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    semantic_meta : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``semantic_meta``.
    mapping_payload : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``mapping_payload``.
    index_path : Path
        Describe ``index_path``.
    documents : SearchDocument
        Describe ``documents``.
    candidate_limit : int
        Describe ``candidate_limit``.
    k : int
        Describe ``k``.
    candidate_ids : set[str]
        Describe ``candidate_ids``.
    row_to_document : int | SearchDocument
        Describe ``row_to_document``.
"""

    semantic_meta: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]]
    mapping_payload: Mapping[
        str, str | int | float | bool | None | list[object] | dict[str, object]
    ]
    index_path: Path
    documents: Sequence[SearchDocument]
    candidate_limit: int
    k: int
    candidate_ids: set[str]
    row_to_document: Mapping[int, SearchDocument]


_FAISS_ENV_OVERRIDE = "KGF_FAISS_MODULE"
_FAISS_FALLBACK_FLAG = "KGF_DISABLE_FAISS_FALLBACK"
_FAISS_DEFAULT_MODULES: tuple[str, ...] = ("faiss", "faiss_cpu")


class _SimpleFaissIndex(FaissIndexProtocol):
    """Lightweight FAISS-like index used when the real library is unavailable.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    dimension : int
        Describe ``dimension``.
"""

    def __init__(self, dimension: int) -> None:
        """Document   init  .

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        dimension : int
            Configure the dimension.
"""
        self.dimension = dimension
        self._vectors: VectorArray = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: VectorArray) -> None:
        """Document add.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Configure the vectors.

        Raises
        ------
        AgentCatalogSearchError
            Raised when message.
"""
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != EMBEDDING_MATRIX_RANK or array.shape[1] != self.dimension:
            message = "Vector dimension does not match index configuration"
            raise AgentCatalogSearchError(message)
        if self._vectors.size == 0:
            self._vectors = np.ascontiguousarray(array)
        else:
            self._vectors = np.vstack((self._vectors, np.ascontiguousarray(array)))

    def train(self, vectors: VectorArray) -> None:
        """Train the index (no-op for simple flat index).

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Describe ``vectors``.
"""
        # Simple flat index doesn't require training

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:  # noqa: ARG002
        """Add vectors with explicit IDs (not supported by simple index).

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Describe ``vectors``.
        ids : tuple[int, ...] | np.int64
            Describe ``ids``.
"""
        # Simple index doesn't support ID mapping, fall back to regular add
        self.add(vectors)

    def search(
        self, vectors: VectorArray, k: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Document search.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Configure the vectors.
        k : int
            Configure the k.

        Returns
        -------
        tuple[tuple[int, ...] | np.float32, tuple[int, ...] | np.int64]
            Describe return value.

        Raises
        ------
        AgentCatalogSearchError
            Raised when message.
"""
        queries = np.asarray(vectors, dtype=np.float32)
        if queries.ndim != EMBEDDING_MATRIX_RANK or queries.shape[1] != self.dimension:
            message = "Query vector dimension does not match index configuration"
            raise AgentCatalogSearchError(message)
        query_count = queries.shape[0]
        distances = np.zeros((query_count, k), dtype=np.float32)
        indices = -np.ones((query_count, k), dtype=np.int64)
        if self._vectors.size == 0:
            return distances, indices
        similarity = np.matmul(queries, self._vectors.T)  # type: ignore[misc]
        if similarity.ndim != EMBEDDING_MATRIX_RANK:  # type: ignore[misc]
            message = "Unexpected similarity matrix shape"
            raise AgentCatalogSearchError(message)
        top_k = min(k, similarity.shape[1])  # type: ignore[misc]
        order = np.argpartition(similarity, -top_k, axis=1)  # type: ignore[misc]
        top_indices = order[:, -top_k:]  # type: ignore[misc,index]
        top_scores = np.take_along_axis(similarity, top_indices, axis=1)  # type: ignore[misc]
        sorted_order = np.argsort(top_scores, axis=1)[:, ::-1]  # type: ignore[misc]
        distances[:, :top_k] = np.take_along_axis(top_scores, sorted_order, axis=1).astype(  # type: ignore[misc]
            np.float32
        )
        indices[:, :top_k] = np.take_along_axis(top_indices, sorted_order, axis=1).astype(np.int64)  # type: ignore[misc]
        return distances, indices


class _SimpleFaissModule:
    """Minimal FAISS module shim using NumPy for tests and local runs.

    <!-- auto:docstring-builder v1 -->

    Implements FaissModuleProtocol for compatibility with FAISS adapters.

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    # FAISS metric constants
    METRIC_INNER_PRODUCT: int = 1
    METRIC_L2: int = 0

    @staticmethod
    def IndexFlatIP(dimension: int) -> FaissIndexProtocol:  # noqa: N802
        """Create a flat inner-product index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        dimension : int
            Vector dimension.

        Returns
        -------
        FaissIndexProtocol
            Flat index instance.
"""
        return cast(FaissIndexProtocol, _SimpleFaissIndex(dimension))

    @staticmethod
    def index_factory(dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        """Create an index from a factory string.

        <!-- auto:docstring-builder v1 -->

        For the simple implementation, factory strings are ignored and a flat index
        is always returned.

        Parameters
        ----------
        dimension : int
            Vector dimension.
        factory_string : str
            Factory description (ignored in simple implementation).
        metric : int
            Metric type (METRIC_INNER_PRODUCT or METRIC_L2) (ignored in simple implementation).

        Returns
        -------
        FaissIndexProtocol
            Flat index instance.
"""
        # Simple implementation ignores factory_string and always returns flat index
        return cast(FaissIndexProtocol, _SimpleFaissIndex(dimension))

    @staticmethod
    def IndexIDMap2(index: FaissIndexProtocol) -> FaissIndexProtocol:  # noqa: N802
        """Wrap an index with 64-bit ID mapping.

        <!-- auto:docstring-builder v1 -->

        For the simple implementation, this is a no-op (returns the index as-is).

        Parameters
        ----------
        index : FaissIndexProtocol
            Base index to wrap.

        Returns
        -------
        FaissIndexProtocol
            Index with ID mapping (same instance in simple implementation).
"""
        # Simple implementation doesn't support ID mapping, return as-is
        return index

    @staticmethod
    def write_index(index: FaissIndexProtocol, path: str) -> None:
        """Persist an index to disk.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        index : FaissIndexProtocol
            Index instance to save.
        path : str
            File path for the persisted index.
"""
        if not isinstance(index, _SimpleFaissIndex):
            message = f"Simple module can only write _SimpleFaissIndex instances, got {type(index)}"
            raise AgentCatalogSearchError(message)
        payload = {
            "dimension": index.dimension,
            "vectors": index._vectors,
        }
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def read_index(path: str) -> FaissIndexProtocol:
        """Load an index from disk.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        path : str
            File path to the persisted index.

        Returns
        -------
        FaissIndexProtocol
            Loaded index instance.
"""
        with Path(path).open("rb") as handle:
            payload_raw: object = pickle.load(handle)  # noqa: S301 - local trusted artifact
        if not isinstance(payload_raw, dict):
            message = "Stored semantic index has invalid payload format"
            raise AgentCatalogSearchError(message)
        payload: dict[str, object] = payload_raw
        vectors_payload = payload.get("vectors", [])
        vectors = np.asarray(vectors_payload, dtype=np.float32)
        dimension_raw = payload.get("dimension")
        dimension: int
        if isinstance(dimension_raw, (int, float)):
            dimension = int(dimension_raw)
        else:
            dimension = int(vectors.shape[1]) if vectors.ndim == EMBEDDING_MATRIX_RANK else 0
        if dimension <= 0:
            message = "Stored semantic index dimension is invalid"
            raise AgentCatalogSearchError(message)
        fallback = _SimpleFaissIndex(dimension)
        if vectors.size:
            if vectors.ndim != EMBEDDING_MATRIX_RANK:
                message = "Stored semantic index vectors have an unexpected shape"
                raise AgentCatalogSearchError(message)
            fallback._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        return cast(FaissIndexProtocol, fallback)

    @staticmethod
    def normalize_L2(vectors: VectorArray) -> None:  # noqa: N802
        """Normalize vectors to unit length in-place.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Array to normalize (modified in-place).
"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # type: ignore[misc]
        norms[norms == 0] = 1.0  # type: ignore[misc]
        vectors /= norms  # type: ignore[misc]


@cache  # type: ignore[misc]
def _simple_faiss_module() -> FaissModuleProtocol:
    """Return a cached NumPy-based FAISS shim for local usage.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    FaissModuleProtocol
        Describe return value.
"""
    return cast(FaissModuleProtocol, _SimpleFaissModule())


def load_faiss(purpose: str) -> FaissModuleProtocol:
    """Import a FAISS module or fall back to the NumPy implementation.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    purpose : str
        Describe ``purpose``.

    Returns
    -------
    FaissModuleProtocol
        Describe return value.
"""
    override = os.getenv(_FAISS_ENV_OVERRIDE)
    candidates: tuple[str, ...] = (override,) if override else _FAISS_DEFAULT_MODULES
    failures: list[str] = []
    for module_name in candidates:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"builtin type SwigPy\w* has no __module__ attribute",
                    category=DeprecationWarning,
                    module=r"faiss.*",
                )
                module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - runtime guard
            failures.append(f"{module_name}: {exc}")
            continue
        if hasattr(module, "IndexFlatIP") and hasattr(module, "write_index"):
            logger.debug(
                "Using FAISS module '%s' for %s",
                module_name,
                purpose,
                extra={
                    "operation": "load_faiss",
                    "status": "success",
                    "module_name": module_name,
                    "purpose": purpose,
                },
            )
            return cast(FaissModuleProtocol, module)
        failures.append(f"{module_name}: missing required attributes")

    failure_text = "; ".join(failures) if failures else "no candidates attempted"
    if os.getenv(_FAISS_FALLBACK_FLAG) == "1":
        message = (
            "FAISS is required to {purpose} but no compatible module could be imported "
            f"(attempted: {failure_text})"
        ).format(purpose=purpose)
        raise AgentCatalogSearchError(message, context={"purpose": purpose, "failures": failures})

    logger.warning(
        "Falling back to simple FAISS module while attempting to %s (reasons: %s)",
        purpose,
        failure_text,
        extra={
            "operation": "load_faiss",
            "status": "warning",
            "purpose": purpose,
            "failures": failures,
        },
    )
    return _simple_faiss_module()


def _tokenize(text: str) -> list[str]:
    """Return normalized word tokens for the given text.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    text : str
        Describe ``text``.

    Returns
    -------
    list[str]
        Describe return value.
"""
    return WORD_PATTERN.findall(text.lower())  # type: ignore[misc]


def _stringify(value: object) -> str | None:
    """Return ``value`` as ``str`` when it is not ``None``.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    value : object
        Describe ``value``.

    Returns
    -------
    str | NoneType
        Describe return value.
"""
    if value is None:
        return None
    return str(value)


def _extract_agent_hints_payload(
    symbol: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
) -> tuple[list[str], list[str]]:
    """Document  extract agent hints payload.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    symbol : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Configure the symbol.

    Returns
    -------
    tuple[list[str], list[str]]
        Describe return value.
"""
    intent_tags: list[str] = []
    tests_to_run: list[str] = []
    agent_hints = symbol.get("agent_hints")
    if isinstance(agent_hints, Mapping):
        raw_tags = agent_hints.get("intent_tags")
        if isinstance(raw_tags, list):
            intent_tags = [str(tag) for tag in raw_tags if tag]
        raw_tests = agent_hints.get("tests_to_run")
        if isinstance(raw_tests, list):
            tests_to_run = [str(test) for test in raw_tests if test]
    return intent_tags, tests_to_run


def _extract_docfacts_text(
    docfacts: Mapping[str, str | int | float | bool | None] | None,
) -> tuple[str | None, str | None]:
    """Document  extract docfacts text.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    docfacts : str | str | int | float | bool | NoneType | NoneType
        Indicate whether docfacts.

    Returns
    -------
    tuple[str | NoneType, str | NoneType]
        Describe return value.
"""
    summary = None
    docstring = None
    if isinstance(docfacts, Mapping):
        summary_val = docfacts.get("summary")
        docstring_val = docfacts.get("docstring")
        summary = _stringify(summary_val)
        docstring = _stringify(docstring_val)
    return summary, docstring


def _extract_anchor_lines(
    symbol: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
) -> tuple[int | None, int | None]:
    """Document  extract anchor lines.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    symbol : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Configure the symbol.

    Returns
    -------
    tuple[int | NoneType, int | NoneType]
        Describe return value.
"""
    anchors = symbol.get("anchors")
    start_line: int | None = None
    end_line: int | None = None
    if isinstance(anchors, Mapping):
        raw_start = anchors.get("start_line")
        raw_end = anchors.get("end_line")
        if isinstance(raw_start, int):
            start_line = raw_start
        if isinstance(raw_end, int):
            end_line = raw_end
    return start_line, end_line


def build_document_from_payload(
    package_name: str,
    module_name: str,
    symbol: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
    symbol_id: str,
    row: int,
) -> SearchDocument:
    """Create a ``SearchDocument`` from the raw symbol payload.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    package_name : str
        Describe ``package_name``.
    module_name : str
        Describe ``module_name``.
    symbol : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``symbol``.
    symbol_id : str
        Describe ``symbol_id``.
    row : int
        Describe ``row``.

    Returns
    -------
    SearchDocument
        Describe return value.
"""
    docfacts_payload = symbol.get("docfacts")
    summary, docstring = _extract_docfacts_text(
        docfacts_payload if isinstance(docfacts_payload, Mapping) else None  # type: ignore[arg-type]
    )
    intent_tags, tests_to_run = _extract_agent_hints_payload(symbol)
    qname_value = _stringify(symbol.get("qname")) or symbol_id
    text_parts = [
        qname_value,
        module_name,
        package_name,
        summary or "",
        docstring or "",
        " ".join(intent_tags),
        " ".join(tests_to_run),
    ]
    normalized_text = " ".join(part for part in text_parts if part)
    tokens = collections.Counter(_tokenize(normalized_text))
    start_line, end_line = _extract_anchor_lines(symbol)
    metrics = symbol.get("metrics")
    stability = None
    deprecated = False
    if isinstance(metrics, Mapping):
        stability = _stringify(metrics.get("stability"))
        deprecated = bool(metrics.get("deprecated"))
    return SearchDocument(
        symbol_id=symbol_id,
        package=package_name,
        module=module_name,
        qname=qname_value,
        kind=str(symbol.get("kind", "object")),
        stability=stability,
        deprecated=deprecated,
        summary=summary,
        docstring=docstring,
        anchor_start=start_line,
        anchor_end=end_line,
        text=normalized_text,
        tokens=tokens,
        row=row,
    )


def iter_symbol_entries(
    catalog: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
) -> Sequence[
    tuple[
        str, str, Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]]
    ]
]:
    """Yield ``(package, module, symbol)`` triples from the catalog payload.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``catalog``.

    Returns
    -------
    tuple[str, str, str | str | int | float | bool | NoneType | list[object] | dict[str, object]]
        Describe return value.
"""
    packages = catalog.get("packages")
    entries: list[
        tuple[
            str,
            str,
            Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
        ]
    ] = []
    if isinstance(packages, list):
        for package in packages:
            if not isinstance(package, Mapping):
                continue
            package_name = _stringify(package.get("name"))
            modules = package.get("modules")
            if not package_name or not isinstance(modules, list):
                continue
            for module in modules:
                if not isinstance(module, Mapping):
                    continue
                module_name = _stringify(module.get("qualified"))
                symbols = module.get("symbols")
                if not module_name or not isinstance(symbols, list):
                    continue
                for symbol in symbols:
                    if isinstance(symbol, Mapping):
                        entries.append((package_name, module_name, symbol))
    return entries


def documents_from_catalog(
    catalog: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
    row_lookup: Mapping[str, int] | None = None,
) -> list[SearchDocument]:
    """Return search documents extracted from the catalog payload.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``catalog``.
    row_lookup : str | int | NoneType, optional
        Describe ``row_lookup``.
        Defaults to ``None``.

    Returns
    -------
    list[SearchDocument]
        Describe return value.
"""
    documents: list[SearchDocument] = []
    for package_name, module_name, symbol in iter_symbol_entries(catalog):
        symbol_id = symbol.get("symbol_id")
        if not isinstance(symbol_id, str):
            continue
        row = row_lookup.get(symbol_id) if row_lookup is not None else -1
        if row_lookup is not None and row is None:
            continue
        documents.append(
            build_document_from_payload(
                package_name=package_name,
                module_name=module_name,
                symbol=symbol,
                symbol_id=symbol_id,
                row=row or -1,
            )
        )
    return documents


def parse_bool(value: str) -> bool | None:
    """Interpret ``value`` as a boolean flag when possible.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    value : str
        Describe ``value``.

    Returns
    -------
    bool | NoneType
        Describe return value.
"""
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def document_matches_facets(document: SearchDocument, facets: Mapping[str, str]) -> bool:
    """Return ``True`` when ``document`` satisfies the provided facets.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    document : SearchDocument
        Describe ``document``.
    facets : str | str
        Describe ``facets``.

    Returns
    -------
    bool
        Describe return value.
"""
    for key, value in facets.items():
        if key == "package" and document.package != value:
            return False
        if key == "module" and document.module != value:
            return False
        if key == "kind" and document.kind != value:
            return False
        if key == "stability" and (document.stability or "") != value:
            return False
        if key == "deprecated":
            parsed = parse_bool(value)
            if parsed is not None and document.deprecated != parsed:
                return False
    return True


def prepare_query_tokens(query: str) -> collections.Counter[str]:
    """Return lexical tokens for ``query`` (with a simple fallback).

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    query : str
        Describe ``query``.

    Returns
    -------
    str
        Describe return value.
"""
    tokens = collections.Counter(_tokenize(query))
    if not tokens:
        tokens = collections.Counter(query.lower().split())
    return tokens


def resolve_search_parameters(
    catalog_search: Mapping[str, str | int | float | bool | None],
    options: SearchOptions,
    document_count: int,
    k: int,
) -> tuple[float, int]:
    """Derive the alpha weight and candidate pool size for search.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog_search : str | str | int | float | bool | NoneType
        Describe ``catalog_search``.
    options : SearchOptions
        Describe ``options``.
    document_count : int
        Describe ``document_count``.
    k : int
        Describe ``k``.

    Returns
    -------
    tuple[float, int]
        Describe return value.
"""
    alpha_value = options.alpha
    if alpha_value is None:
        alpha_candidate = catalog_search.get("alpha")
        if isinstance(alpha_candidate, (int, float)):
            alpha_value = float(alpha_candidate)
    if alpha_value is None:
        alpha_value = 0.6
    alpha_value = min(max(alpha_value, 0.0), 1.0)

    candidate_value = options.candidate_pool
    if candidate_value is None:
        pool_value = catalog_search.get("candidate_pool")
        if isinstance(pool_value, int) and pool_value > 0:
            candidate_value = pool_value
    if candidate_value is None:
        candidate_value = document_count
    candidate_limit = min(max(candidate_value, k), document_count)
    return alpha_value, candidate_limit


def compute_lexical_scores(
    query_tokens: collections.Counter[str],
    documents: Sequence[SearchDocument],
    query: str,
) -> dict[str, float]:
    """Return lexical similarity scores for the candidate documents.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    query_tokens : str
        Describe ``query_tokens``.
    documents : SearchDocument
        Describe ``documents``.
    query : str
        Describe ``query``.

    Returns
    -------
    dict[str, float]
        Describe return value.
"""
    scores: dict[str, float] = {}
    lowered_query = query.lower()
    for document in documents:
        score = 0.0
        for token, weight in query_tokens.items():
            token_count = document.tokens.get(token, 0)
            if token_count:
                score += float(min(weight, token_count))
        if score == 0.0 and lowered_query in document.qname.lower():
            score = 1.0
        scores[document.symbol_id] = score
    return scores


def select_lexical_candidates(
    lexical_scores: Mapping[str, float],
    documents: Sequence[SearchDocument],
    candidate_limit: int,
) -> list[SearchDocument]:
    """Return the highest-scoring lexical candidates.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    lexical_scores : str | float
        Describe ``lexical_scores``.
    documents : SearchDocument
        Describe ``documents``.
    candidate_limit : int
        Describe ``candidate_limit``.

    Returns
    -------
    list[SearchDocument]
        Describe return value.
"""

    def _get_score(doc: SearchDocument) -> float:
        return lexical_scores.get(doc.symbol_id, 0.0)

    sorted_docs = sorted(
        documents,
        key=_get_score,
        reverse=True,
    )
    return sorted_docs[:candidate_limit]


def _resolve_semantic_index_metadata(
    catalog: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
    repo_root: Path,
) -> (
    tuple[
        Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]], Path, Path
    ]
    | None
):
    """Return semantic index metadata when available, verifying artifacts.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``catalog``.
    repo_root : Path
        Describe ``repo_root``.

    Returns
    -------
    tuple[str | str | int | float | bool | NoneType | list[object] | dict[str, object], Path, Path] | NoneType
        Describe return value.
"""
    semantic_meta = catalog.get("semantic_index")
    if not isinstance(semantic_meta, Mapping):
        return None
    index_rel = semantic_meta.get("index")
    mapping_rel = semantic_meta.get("mapping")
    if not isinstance(index_rel, str) or not isinstance(mapping_rel, str):
        message = "Semantic index metadata is missing artifact paths"
        raise AgentCatalogSearchError(message)
    index_path = (repo_root / index_rel).resolve(strict=True)
    mapping_path = (repo_root / mapping_rel).resolve(strict=True)
    # Validate paths stay under repo_root to prevent directory traversal
    try:
        index_path.relative_to(repo_root.resolve(strict=True))
        mapping_path.relative_to(repo_root.resolve(strict=True))
    except ValueError as exc:
        message = "Semantic index artifact paths must be under repo_root"
        raise AgentCatalogSearchError(message, cause=exc) from exc
    if not index_path.exists() or not mapping_path.exists():
        message = "Semantic index artifacts are missing from disk"
        raise AgentCatalogSearchError(
            message, context={"index_path": str(index_path), "mapping_path": str(mapping_path)}
        )
    # Type cast for return - semantic_meta is validated as Mapping above
    semantic_meta_typed: Mapping[
        str, str | int | float | bool | None | list[object] | dict[str, object]
    ] = cast(
        Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
        semantic_meta,
    )
    return semantic_meta_typed, index_path, mapping_path


def _load_row_lookup(
    mapping_path: Path,
) -> tuple[
    dict[str, int], Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]]
]:
    """Return the row lookup mapping and raw payload from ``mapping_path``.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    mapping_path : Path
        Describe ``mapping_path``.

    Returns
    -------
    tuple[dict[str, int], str | str | int | float | bool | NoneType | list[object] | dict[str, object]]
        Describe return value.
"""
    mapping_payload_raw = json.loads(mapping_path.read_text(encoding="utf-8"))  # type: ignore[misc]
    if not isinstance(mapping_payload_raw, dict):  # type: ignore[misc]
        message = "Semantic mapping file does not contain valid JSON object"
        raise AgentCatalogSearchError(message)
    # Cast to expected type - JSON can have broader types
    mapping_payload: Mapping[
        str, str | int | float | bool | None | list[object] | dict[str, object]
    ] = cast(
        Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
        mapping_payload_raw,
    )
    symbols_payload = mapping_payload.get("symbols")
    if not isinstance(symbols_payload, list):
        message = "Semantic mapping file does not contain symbols list"
        raise AgentCatalogSearchError(message)
    row_lookup: dict[str, int] = {}
    for entry in symbols_payload:
        if not isinstance(entry, Mapping):
            continue
        symbol_id = entry.get("symbol_id")
        row = entry.get("row")
        if isinstance(symbol_id, str) and isinstance(row, int):
            row_lookup[symbol_id] = row
    return row_lookup, mapping_payload


def _load_sentence_transformer(model_name: str) -> EmbeddingModelProtocol:
    """Instantiate a sentence-transformers model, raising ``AgentCatalogSearchError``.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    model_name : str
        Describe ``model_name``.

    Returns
    -------
    EmbeddingModelProtocol
        Describe return value.
"""
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:  # pragma: no cover - runtime guard
        message = "sentence-transformers is required for semantic search"
        raise AgentCatalogSearchError(message, cause=exc) from exc
    factory_raw = getattr(module, "SentenceTransformer", None)  # type: ignore[misc]
    factory: Callable[[str], object] | None = factory_raw
    if not callable(factory):  # pragma: no cover - defensive guard
        message = "sentence-transformers is missing the SentenceTransformer class"
        raise AgentCatalogSearchError(message)
    try:
        model = factory(model_name)
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Unable to load embedding model '{model_name}'"
        raise AgentCatalogSearchError(
            message, cause=exc, context={"model_name": model_name}
        ) from exc
    return cast(EmbeddingModelProtocol, model)


def _resolve_embedding_model(
    options: SearchOptions, semantic_meta: Mapping[str, str | int | float | bool | None]
) -> tuple[str, EmbeddingModelProtocol]:
    """Return the embedding model name and instance used for vector search.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    options : SearchOptions
        Describe ``options``.
    semantic_meta : str | str | int | float | bool | NoneType
        Describe ``semantic_meta``.

    Returns
    -------
    tuple[str, EmbeddingModelProtocol]
        Describe return value.
"""
    model_name_raw = options.embedding_model or _stringify(semantic_meta.get("model"))
    model_name: str | None = model_name_raw
    if not model_name:
        message = "Semantic index metadata does not include the embedding model name"
        raise AgentCatalogSearchError(message)
    loader = options.model_loader or _load_sentence_transformer
    try:
        model = loader(model_name)
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Unable to load embedding model '{model_name}'"
        raise AgentCatalogSearchError(
            message, cause=exc, context={"model_name": model_name}
        ) from exc
    return model_name, model


def _encode_query(
    model: EmbeddingModelProtocol,
    query: str,
    *,
    batch_size: int,
) -> VectorArray:
    """Return normalized embeddings for ``query`` using ``model``.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    model : EmbeddingModelProtocol
        Describe ``model``.
    query : str
        Describe ``query``.
    batch_size : int
        Describe ``batch_size``.

    Returns
    -------
    tuple[int, ...] | np.float32
        Describe return value.
"""
    try:
        encoded = model.encode(
            [query],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        message = f"Embedding model failed to encode query: {exc}"
        raise AgentCatalogSearchError(message, cause=exc) from exc
    return np.asarray(encoded, dtype=np.float32)


def _scores_from_index(
    distances: VectorArray,
    indices: npt.NDArray[np.int64],
    context: VectorSearchContext,
) -> dict[str, float]:
    """Map FAISS search outputs to candidate symbol scores.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    distances : tuple[int, ...] | np.float32
        Describe ``distances``.
    indices : tuple[int, ...] | np.int64
        Describe ``indices``.
    context : VectorSearchContext
        Describe ``context``.

    Returns
    -------
    dict[str, float]
        Describe return value.
"""
    vector_scores: dict[str, float] = {}
    for idx, distance_row in enumerate(distances):  # type: ignore[misc]
        for rank, score in enumerate(distance_row):  # type: ignore[misc]
            row_index = int(indices[idx, rank])  # type: ignore[misc]
            if row_index < 0:
                continue
            document = context.row_to_document.get(row_index)
            if document is None or document.symbol_id not in context.candidate_ids:
                continue
            vector_scores[document.symbol_id] = float(score)  # type: ignore[misc]
    return vector_scores


def compute_vector_scores(
    query: str,
    options: SearchOptions,
    context: VectorSearchContext,
) -> dict[str, float]:
    """Return vector similarity scores for ``query`` and the candidate set.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    query : str
        Describe ``query``.
    options : SearchOptions
        Describe ``options``.
    context : VectorSearchContext
        Describe ``context``.

    Returns
    -------
    dict[str, float]
        Describe return value.
"""
    _, model = _resolve_embedding_model(options, context.semantic_meta)  # type: ignore[arg-type]
    batch_size = options.batch_size or 32
    query_vector = _encode_query(model, query, batch_size=batch_size)
    faiss_module = load_faiss("perform semantic search")
    index = faiss_module.read_index(str(context.index_path))
    distances, indices = index.search(query_vector, context.k)
    return _scores_from_index(distances, indices, context)


def merge_scores(
    documents: Mapping[str, SearchDocument],
    candidate_ids: set[str],
    lexical_scores: Mapping[str, float],
    vector_scores: Mapping[str, float],
    alpha_value: float,
) -> list[SearchResult]:
    """Combine lexical/vector scores into ranked ``SearchResult`` records.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    documents : str | SearchDocument
        Describe ``documents``.
    candidate_ids : set[str]
        Describe ``candidate_ids``.
    lexical_scores : str | float
        Describe ``lexical_scores``.
    vector_scores : str | float
        Describe ``vector_scores``.
    alpha_value : float
        Describe ``alpha_value``.

    Returns
    -------
    list[SearchResult]
        Describe return value.
"""
    results: list[SearchResult] = []
    max_lexical = max(lexical_scores.get(symbol_id, 0.0) for symbol_id in candidate_ids) or 1.0
    for symbol_id in candidate_ids:
        document = documents[symbol_id]
        lexical_raw = lexical_scores.get(symbol_id, 0.0)
        lexical_norm = lexical_raw / max_lexical if max_lexical > 0 else 0.0
        vector_raw = vector_scores.get(symbol_id, -1.0)
        vector_norm = 0.0
        if vector_raw > -1.0:
            vector_norm = max(min((vector_raw + 1.0) / 2.0, 1.0), 0.0)
        combined = alpha_value * vector_norm + (1.0 - alpha_value) * lexical_norm
        results.append(
            SearchResult(
                symbol_id=symbol_id,
                score=combined,
                lexical_score=lexical_norm,
                vector_score=vector_norm,
                package=document.package,
                module=document.module,
                qname=document.qname,
                kind=document.kind,
                stability=document.stability,
                deprecated=document.deprecated,
                summary=document.summary,
                docstring=document.docstring,
                anchor={"start_line": document.anchor_start, "end_line": document.anchor_end},
            )
        )
    results.sort(
        key=lambda item: (
            item.score,
            item.lexical_score,
            item.vector_score,
            item.qname,
        ),
        reverse=True,
    )
    return results


def _prepare_search_documents(
    catalog: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
    repo_root: Path,
    facets: Mapping[str, str],
) -> tuple[
    list[SearchDocument],
    tuple[
        Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]], Path, Path
    ]
    | None,
    Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]] | None,
]:
    """Prepare and filter documents from catalog.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``catalog``.
    repo_root : Path
        Describe ``repo_root``.
    facets : str | str
        Describe ``facets``.

    Returns
    -------
    tuple[list[SearchDocument], tuple[str | str | int | float | bool | NoneType | list[object] | dict[str, object], Path, Path] | NoneType, str | str | int | float | bool | NoneType | list[object] | dict[str, object] | NoneType]
        Filtered documents, semantic metadata info (if available), and mapping payload.
"""
    semantic_meta_info = _resolve_semantic_index_metadata(catalog, repo_root)
    row_lookup: dict[str, int] | None = None
    mapping_payload: (
        Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]] | None
    ) = None
    if semantic_meta_info is not None:
        _, _, mapping_path = semantic_meta_info
        row_lookup, mapping_payload = _load_row_lookup(mapping_path)

    documents = documents_from_catalog(catalog, row_lookup=row_lookup)
    if facets:
        documents = [doc for doc in documents if document_matches_facets(doc, facets)]
    return documents, semantic_meta_info, mapping_payload


def _perform_lexical_search(
    query: str,
    documents: list[SearchDocument],
    candidate_limit: int,
) -> tuple[dict[str, float], set[str], dict[str, SearchDocument]]:
    """Perform lexical search and return scores, candidate IDs, and document lookup.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    query : str
        Describe ``query``.
    documents : list[SearchDocument]
        Describe ``documents``.
    candidate_limit : int
        Describe ``candidate_limit``.

    Returns
    -------
    tuple[dict[str, float], set[str], dict[str, SearchDocument]]
        Lexical scores, candidate symbol IDs, and document lookup map.
"""
    query_tokens = prepare_query_tokens(query)
    lexical_scores = compute_lexical_scores(query_tokens, documents, query)
    lexical_candidates = select_lexical_candidates(lexical_scores, documents, candidate_limit)
    candidate_ids: set[str] = {doc.symbol_id for doc in lexical_candidates}
    doc_by_id = {doc.symbol_id: doc for doc in documents}
    return lexical_scores, candidate_ids, doc_by_id


@dataclass(slots=True)
class VectorSearchParams:
    """Parameters for vector search operation.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    semantic_meta_info : tuple[str | str | int | float | bool | NoneType | list[object] | dict[str, object], Path, Path]
        Describe ``semantic_meta_info``.
    mapping_payload : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Describe ``mapping_payload``.
    documents : list[SearchDocument]
        Describe ``documents``.
    candidate_limit : int
        Describe ``candidate_limit``.
    k : int
        Describe ``k``.
    candidate_ids : set[str]
        Describe ``candidate_ids``.
"""

    semantic_meta_info: tuple[
        Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]], Path, Path
    ]
    mapping_payload: Mapping[
        str, str | int | float | bool | None | list[object] | dict[str, object]
    ]
    documents: list[SearchDocument]
    candidate_limit: int
    k: int
    candidate_ids: set[str]


def _perform_vector_search(
    query: str,
    options: SearchOptions,
    params: VectorSearchParams,
) -> tuple[dict[str, float], float]:
    """Perform vector search and return scores and alpha value.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    query : str
        Describe ``query``.
    options : SearchOptions
        Describe ``options``.
    params : VectorSearchParams
        Describe ``params``.

    Returns
    -------
    tuple[dict[str, float], float]
        Vector similarity scores and alpha blending value.
"""
    semantic_meta, index_path, _ = params.semantic_meta_info
    row_to_document = {doc.row: doc for doc in params.documents if doc.row >= 0}
    context = VectorSearchContext(
        semantic_meta=semantic_meta,
        mapping_payload=params.mapping_payload,
        index_path=index_path,
        documents=params.documents,
        candidate_limit=params.candidate_limit,
        k=params.k,
        candidate_ids=params.candidate_ids,
        row_to_document=row_to_document,
    )
    try:
        vector_scores = compute_vector_scores(query, options, context)
    except AgentCatalogSearchError:
        return {}, 0.0
    else:
        return vector_scores, 0.0


@dataclass(slots=True)
class SearchRequest:
    """Request parameters for catalog search.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    repo_root : Path
        Describe ``repo_root``.
    query : str
        Describe ``query``.
    k : int, optional
        Describe ``k``.
        Defaults to ``10``.
"""

    repo_root: Path
    query: str
    k: int = 10


def search_catalog(
    catalog: Mapping[str, str | int | float | bool | None | list[object] | dict[str, object]],
    *,
    request: SearchRequest,
    options: SearchOptions | None = None,
    metrics: MetricsProvider | None = None,
) -> list[SearchResult]:
    """Execute hybrid lexical/vector search against the catalog.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    catalog : str | str | int | float | bool | NoneType | list[object] | dict[str, object]
        Catalog payload containing packages, modules, symbols, and optional semantic_index metadata.
    request : SearchRequest
        Search request containing repo_root, query, and k parameters.
    options : SearchOptions | NoneType, optional
        Optional search parameters (alpha, facets, embedding_model, etc.). Defaults to None.
        Defaults to ``None``.
    metrics : MetricsProvider | NoneType, optional
        Metrics provider for recording search duration and counts. Defaults to None (uses default).
        Defaults to ``None``.

    Returns
    -------
    list[SearchResult]
        Ranked search results with combined lexical/vector scores.

    Raises
    ------
    AgentCatalogSearchError
        Raised when search fails due to invalid input, missing artifacts, or model errors.
"""
    active_metrics = metrics or MetricsProvider.default()
    active_options = options or SearchOptions()
    trimmed_query = request.query.strip()
    k = request.k
    if not trimmed_query:
        message = "Search query must not be empty"
        raise AgentCatalogSearchError(message, context={"query": request.query})

    with (
        with_fields(
            logger,
            operation="search_catalog",
            status="started",
            k=k,
            query_length=len(trimmed_query),
        ) as log_adapter,
        observe_duration(active_metrics, "search", component="agent_catalog") as obs,
    ):
        try:
            facets = active_options.facets or {}
            documents, semantic_meta_info, mapping_payload = _prepare_search_documents(
                catalog, request.repo_root, facets
            )
            if not documents:
                log_adapter.info(
                    "Search completed with no documents matching facets",
                    extra={"status": "success", "result_count": 0},
                )
                obs.success()
                return []
            catalog_search = catalog.get("search")
            search_config: Mapping[str, str | int | float | bool | None] = (
                cast(Mapping[str, str | int | float | bool | None], catalog_search)
                if isinstance(catalog_search, Mapping)
                else {}
            )
            alpha_value, candidate_limit = resolve_search_parameters(
                search_config, active_options, len(documents), k
            )

            lexical_scores, candidate_ids, doc_by_id = _perform_lexical_search(
                trimmed_query, documents, candidate_limit
            )
            if not candidate_ids:
                log_adapter.info(
                    "Search completed with no lexical candidates",
                    extra={"status": "success", "result_count": 0},
                )
                obs.success()
                return []
            vector_scores: dict[str, float] = {}
            if semantic_meta_info is not None and mapping_payload is not None:
                params = VectorSearchParams(
                    semantic_meta_info=semantic_meta_info,
                    mapping_payload=mapping_payload,
                    documents=documents,
                    candidate_limit=candidate_limit,
                    k=k,
                    candidate_ids=candidate_ids,
                )
                vector_scores, alpha_value = _perform_vector_search(
                    trimmed_query, active_options, params
                )
            else:
                alpha_value = 0.0

            results = merge_scores(
                doc_by_id, candidate_ids, lexical_scores, vector_scores, alpha_value
            )
            final_results = results[:k]
            obs.success()
            log_adapter.info(
                "Search completed successfully",
                extra={
                    "status": "success",
                    "result_count": len(final_results),
                    "alpha": alpha_value,
                },
            )
        except AgentCatalogSearchError:
            obs.error()
            raise
        except Exception as exc:
            obs.error()
            message = f"Unexpected error during catalog search: {exc}"
            raise AgentCatalogSearchError(message, cause=exc) from exc
        else:
            return final_results
