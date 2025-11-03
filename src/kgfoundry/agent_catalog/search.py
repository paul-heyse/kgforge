"""Hybrid search utilities for the agent catalog.

This module orchestrates lexical and vector search across the catalog payload.
Vector operations rely on the typed helpers in
``kgfoundry_common.numpy_typing`` so that downstream consumers (and mypy) can
reason about the ndarray shapes involved. The resulting scores conform to the
``schema/models/search_result.v1.json`` schema.

Helpers for constructing SearchOptions and SearchDocument ensure consistent
defaults and early validation of parameters and dependencies. These helpers
emit RFC 9457 Problem Details on validation failures.
"""

from __future__ import annotations

import collections
import importlib
import json
import os
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Protocol, TypedDict, Unpack, cast

import numpy as np

from kgfoundry_common.errors import AgentCatalogSearchError
from kgfoundry_common.logging import get_logger
from kgfoundry_common.numpy_typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    topk_indices,
)
from kgfoundry_common.numpy_typing import (
    normalize_l2 as _normalize_l2_array,
)
from kgfoundry_common.observability import MetricsProvider
from orchestration.safe_pickle import dump as safe_pickle_dump
from orchestration.safe_pickle import load as safe_pickle_load
from search_api.types import (
    FaissIndexProtocol,
    FaissModuleProtocol,
    IndexArray,
    VectorArray,
    wrap_faiss_module,
)

logger = get_logger(__name__)

JsonLike = str | int | float | bool | list[object] | dict[str, object] | None
CatalogMapping = Mapping[str, JsonLike]
PrimitiveMapping = Mapping[str, str | int | float | bool | None]

EMBEDDING_MATRIX_RANK = 2
WORD_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9_]+")

# Canonical default values for search options (used by all helpers)
_DEFAULT_ALPHA = 0.6
_DEFAULT_CANDIDATE_POOL = 100
_DEFAULT_BATCH_SIZE = 32

# Allowed facet keys (strict allow-list for early validation)
_ALLOWED_FACET_KEYS = frozenset(["package", "module", "kind", "stability"])


class SearchOptionsPayload(TypedDict, total=False):
    """Typed payload for SearchOptions configuration.

    This TypedDict defines the complete search options with all optional
    fields. It aligns with JSON Schema definitions and ensures type safety.
    """

    alpha: float
    facets: Mapping[str, str]
    candidate_pool: int
    model_loader: Callable[[str], EmbeddingModelProtocol]
    embedding_model: str
    batch_size: int


class SearchOptionsOverrides(TypedDict, total=False):
    """Optional overrides for search option helper factories."""

    alpha: float
    candidate_pool: int
    batch_size: int


class SearchDocumentPayload(TypedDict):
    """Typed payload for SearchDocument construction.

    This TypedDict ensures all required fields are present and properly typed,
    providing parity with the JSON Schema definition for search documents.
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
    row: int


class SearchDocumentOverrides(TypedDict, total=False):
    stability: str | None
    deprecated: bool
    summary: str | None
    docstring: str | None
    anchor_start: int | None
    anchor_end: int | None
    row: int


# Public API exports
__all__ = [
    "EmbeddingModelProtocol",
    "MetricsProvider",
    "PreparedSearchArtifacts",
    "SearchConfig",
    "SearchDocument",
    "SearchDocumentPayload",
    "SearchOptions",
    "SearchOptionsPayload",
    "SearchRequest",
    "SearchResult",
    "VectorSearchContext",
    "build_default_search_options",
    "build_embedding_aware_search_options",
    "build_faceted_search_options",
    "documents_from_catalog",
    "make_search_document",
    "search_catalog",
]


@dataclass(slots=True)
class SearchRequest:
    """Request parameters for searching the agent catalog.

    Parameters
    ----------
    repo_root : Path
        Repository root for resolving semantic index artifacts.
    query : str
        Search query text.
    k : int
        Number of results to return.
    """

    repo_root: Path
    query: str
    k: int


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


def build_default_search_options(
    *,
    alpha: float | None = None,
    candidate_pool: int | None = None,
    batch_size: int | None = None,
    embedding_model: str | None = None,
    model_loader: Callable[[str], EmbeddingModelProtocol] | None = None,
) -> SearchOptions:
    """Build default SearchOptions with validated parameters.

    This helper factory ensures consistent defaults across CLI, client, and
    docs tooling. All parameters are optional and default to canonical values.

    Parameters
    ----------
    alpha : float | None, optional
        Hybrid search weighting (0.0 lexical, 1.0 vector).
        Defaults to 0.6 if not provided.
    candidate_pool : int | None, optional
        Pre-filtering candidate pool size. Must be >= k.
        Defaults to 100 if not provided.
    batch_size : int | None, optional
        Embedding batch size for model encoding.
        Defaults to 32 if not provided.
    embedding_model : str | None, optional
        Name/path of embedding model.
    model_loader : Callable | None, optional
        Factory function to load embedding models.

    Returns
    -------
    SearchOptions
        Fully populated options with validated defaults.

    Raises
    ------
    AgentCatalogSearchError
        If alpha is outside [0.0, 1.0] or candidate_pool is negative.

    Examples
    --------
    Create search options with default values (all defaults applied):

    >>> opts = build_default_search_options()
    >>> assert opts.alpha == 0.6  # default alpha
    >>> assert opts.candidate_pool == 100  # default pool
    >>> assert opts.batch_size == 32  # default batch

    Override specific parameters while keeping others as defaults:

    >>> opts = build_default_search_options(alpha=0.5, candidate_pool=500)
    >>> assert opts.alpha == 0.5  # explicit override
    >>> assert opts.candidate_pool == 500  # explicit override
    >>> assert opts.batch_size == 32  # still default

    Invalid alpha (outside [0.0, 1.0]) raises AgentCatalogSearchError:

    >>> try:  # doctest: +SKIP (requires full env)
    ...     build_default_search_options(alpha=1.5)
    ... except Exception as e:
    ...     assert "alpha" in str(e).lower()
    """
    final_alpha = alpha if alpha is not None else _DEFAULT_ALPHA
    final_candidate_pool = candidate_pool if candidate_pool is not None else _DEFAULT_CANDIDATE_POOL
    final_batch_size = batch_size if batch_size is not None else _DEFAULT_BATCH_SIZE

    # Validate alpha range
    if not 0.0 <= final_alpha <= 1.0:
        message = f"alpha must be in [0.0, 1.0], got {final_alpha}"
        raise AgentCatalogSearchError(
            message,
            context={"alpha": final_alpha},
        )

    # Validate candidate pool
    if final_candidate_pool < 0:
        message = f"candidate_pool must be non-negative, got {final_candidate_pool}"
        raise AgentCatalogSearchError(
            message,
            context={"candidate_pool": final_candidate_pool},
        )

    return SearchOptions(
        alpha=final_alpha,
        candidate_pool=final_candidate_pool,
        batch_size=final_batch_size,
        embedding_model=embedding_model,
        model_loader=model_loader,
    )


def _validate_facets(facets: Mapping[str, str]) -> None:
    invalid_keys = set(facets.keys()) - _ALLOWED_FACET_KEYS
    if not invalid_keys:
        return
    message = f"Invalid facet keys: {sorted(invalid_keys)}. Allowed: {sorted(_ALLOWED_FACET_KEYS)}"
    raise AgentCatalogSearchError(
        message,
        context={
            "invalid_keys": sorted(invalid_keys),
            "allowed_keys": sorted(_ALLOWED_FACET_KEYS),
        },
    )


def build_faceted_search_options(
    *,
    facets: Mapping[str, str],
    **overrides: Unpack[SearchOptionsOverrides],
) -> SearchOptions:
    """Build `SearchOptions` with strict facet validation.

    Parameters
    ----------
    facets : Mapping[str, str]
        Facet filters (package, module, kind, stability).
    **overrides : SearchOptionsOverrides
        Optional overrides forwarded to :func:`build_default_search_options`.
        Supports ``alpha``, ``candidate_pool``, ``batch_size``, ``embedding_model``,
        and ``model_loader`` keyword arguments.

    Returns
    -------
    SearchOptions
        Validated options with facets included.

    Raises
    ------
    AgentCatalogSearchError
        If any facet key is not in the allow-list.
    """
    # Validate facet keys against allow-list
    _validate_facets(facets)

    opts = build_default_search_options(**overrides)
    opts.facets = dict(facets)
    return opts


def build_embedding_aware_search_options(
    *,
    embedding_model: str,
    model_loader: Callable[[str], EmbeddingModelProtocol],
    facets: Mapping[str, str] | None = None,
    **overrides: Unpack[SearchOptionsOverrides],
) -> SearchOptions:
    """Build `SearchOptions` with explicit embedding configuration.

    Parameters
    ----------
    embedding_model : str
        Embedding model identifier (required).
    model_loader : Callable
        Factory callable that loads the embedding model (required).
    facets : Mapping[str, str] | None, optional
        Optional facet filters; validated when provided.
    **overrides : SearchOptionsOverrides
        Additional overrides forwarded to :func:`build_default_search_options`.
        Useful for setting ``alpha``, ``candidate_pool``, or ``batch_size``.

    Returns
    -------
    SearchOptions
        Validated options with embedding awareness configured.
    """
    options = build_default_search_options(
        embedding_model=embedding_model,
        model_loader=model_loader,
        **overrides,
    )
    if facets is not None:
        _validate_facets(facets)
        options.facets = dict(facets)
    return options


def make_search_document(
    *,
    symbol_id: str,
    package: str,
    module: str,
    qname: str,
    kind: str,
    **overrides: Unpack[SearchDocumentOverrides],
) -> SearchDocument:
    """Create a validated SearchDocument with all required fields populated.

    This helper centralizes SearchDocument construction, ensuring consistent
    normalization (whitespace stripping, deterministic token ordering) and
    providing a typed, documented API for all callers.

    Parameters
    ----------
    symbol_id : str
        Unique symbol identifier (required).
    package : str
        Package name (required).
    module : str
        Module qualified name (required).
    qname : str
        Fully qualified symbol name (required).
    kind : str
        Symbol kind (e.g., "class", "function", "module") (required).
    stability : str | None, optional
        Stability level (e.g., "stable", "experimental"). Defaults to None.
    deprecated : bool, optional
        Whether the symbol is deprecated. Defaults to False.
    overrides : SearchDocumentOverrides, optional
        Additional fields such as ``summary``, ``docstring``, ``stability``,
        ``deprecated``, anchor positions, and row index. Defaults applied when
        keys are omitted.

    Returns
    -------
    SearchDocument
        Normalized and validated search document.

    Examples
    --------
    >>> doc = make_search_document(
    ...     symbol_id="py:kgfoundry.search.find_similar",
    ...     package="kgfoundry",
    ...     module="kgfoundry.agent_catalog.search",
    ...     qname="find_similar",
    ...     kind="function",
    ...     summary="Find similar symbols in catalog",
    ... )
    >>> assert doc.symbol_id == "py:kgfoundry.search.find_similar"
    >>> assert doc.package == "kgfoundry"
    """
    # Normalize and strip whitespace from string fields
    summary_value = overrides.get("summary")
    docstring_value = overrides.get("docstring")
    normalized_summary = summary_value.strip() if summary_value else None
    normalized_docstring = docstring_value.strip() if docstring_value else None
    normalized_qname = qname.strip()

    # Build text for lexical search: concatenate normalized fields
    text_parts = [
        normalized_qname,
        module.strip(),
        package.strip(),
        normalized_summary or "",
        normalized_docstring or "",
    ]
    normalized_text = " ".join(part for part in text_parts if part)

    # Tokenize the normalized text for lexical scoring
    tokens = collections.Counter(_tokenize(normalized_text))

    return SearchDocument(
        symbol_id=symbol_id,
        package=package,
        module=module,
        qname=normalized_qname,
        kind=kind,
        stability=overrides.get("stability"),
        deprecated=overrides.get("deprecated", False),
        summary=normalized_summary,
        docstring=normalized_docstring,
        anchor_start=overrides.get("anchor_start"),
        anchor_end=overrides.get("anchor_end"),
        text=normalized_text,
        tokens=tokens,
        row=overrides.get("row", -1),
    )


LEXICAL_FIELDS = [
    "qname",
    "module",
    "package",
    "summary",
    "docstring",
    "agent_hints.intent_tags",
]


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
    """Supporting data required to execute a vector search."""

    semantic_meta: CatalogMapping
    mapping_payload: CatalogMapping
    index_path: Path
    documents: Sequence[SearchDocument]
    candidate_limit: int
    k: int
    candidate_ids: set[str]
    row_to_document: Mapping[int, SearchDocument]


@dataclass(slots=True)
class _VectorSearchInputs:
    """Resolved components needed to perform FAISS vector search."""

    model: EmbeddingModelProtocol
    batch_size: int
    index: FaissIndexProtocol
    candidate_limit: int
    row_lookup: Mapping[int, SearchDocument]


@dataclass(slots=True)
class PreparedSearchArtifacts:
    """Documents and optional semantic metadata extracted from the catalog."""

    documents: list[SearchDocument]
    semantic_meta: tuple[CatalogMapping, Path, Path] | None
    mapping_payload: CatalogMapping | None


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
        self._vectors: FloatMatrix = np.empty((0, dimension), dtype=np.float32, order="C")

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
        array = np.asarray(vectors, dtype=np.float32, order="C")
        if array.ndim != EMBEDDING_MATRIX_RANK or array.shape[1] != self.dimension:
            message = "Vector dimension does not match index configuration"
            raise AgentCatalogSearchError(message)
        self._vectors = (
            array
            if self._vectors.size == 0
            else np.vstack((self._vectors, array.astype(np.float32, copy=False)))
        )

    def train(self, vectors: VectorArray) -> None:
        """Train the index (no-op for simple flat index).

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Describe ``vectors``.
        """
        # Simple flat index doesn't require training

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:
        """Add vectors with explicit IDs (not supported by simple index).

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Describe ``vectors``.
        ids : tuple[int, ...] | np.int64
            Describe ``ids``.
        """
        id_array = np.asarray(ids, dtype=np.int64)
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.shape[0] != id_array.shape[0]:
            message = "IDs length must match vector count"
            raise AgentCatalogSearchError(message)
        # Simple index doesn't support ID mapping, fall back to regular add
        self.add(vector_array)

    def search(self, vectors: VectorArray, k: int) -> tuple[FloatMatrix, IntVector]:
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
        queries: FloatMatrix = np.asarray(vectors, dtype=np.float32, order="C")
        if queries.ndim != EMBEDDING_MATRIX_RANK or queries.shape[1] != self.dimension:
            message = "Query vector dimension does not match index configuration"
            raise AgentCatalogSearchError(message)
        query_count = queries.shape[0]
        distances: FloatMatrix = np.zeros((query_count, k), dtype=np.float32)
        indices: IntVector = -np.ones((query_count, k), dtype=np.int64)
        if self._vectors.size == 0:
            return distances, indices
        similarity_matrix: FloatMatrix = cast(FloatMatrix, queries @ self._vectors.T)
        if similarity_matrix.ndim != EMBEDDING_MATRIX_RANK:
            message = "Unexpected similarity matrix shape"
            raise AgentCatalogSearchError(message)
        top_k = min(k, similarity_matrix.shape[1])
        for row_idx in range(similarity_matrix.shape[0]):
            scores_row = cast(FloatVector, similarity_matrix[row_idx])
            top_indices = topk_indices(scores_row, top_k)
            distances[row_idx, :top_k] = scores_row[top_indices]
            indices[row_idx, :top_k] = top_indices
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
    metric_inner_product: int = 1
    metric_l2: int = 0

    @staticmethod
    def _create_flat_index(dimension: int) -> _SimpleFaissIndex:
        if dimension <= 0:
            message = f"Index dimension must be positive, got {dimension}"
            raise AgentCatalogSearchError(message)
        return _SimpleFaissIndex(dimension)

    @staticmethod
    def _ensure_simple_index(index: FaissIndexProtocol) -> _SimpleFaissIndex:
        if isinstance(index, _SimpleFaissIndex):
            return index
        message = (
            f"Simple module can only operate on _SimpleFaissIndex instances, got {type(index)}"
        )
        raise AgentCatalogSearchError(message)

    @staticmethod
    def index_flat_ip(dimension: int) -> FaissIndexProtocol:
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
        return cast(FaissIndexProtocol, _SimpleFaissModule._create_flat_index(dimension))

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
        del factory_string, metric
        # Simple implementation ignores factory configuration and always returns flat index
        return cast(FaissIndexProtocol, _SimpleFaissModule._create_flat_index(dimension))

    @staticmethod
    def index_id_map2(index: FaissIndexProtocol) -> FaissIndexProtocol:
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
        return cast(FaissIndexProtocol, _SimpleFaissModule._ensure_simple_index(index))

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
        simple_index = _SimpleFaissModule._ensure_simple_index(index)
        vectors_payload = cast(list[list[float]], simple_index._vectors.tolist())
        payload: dict[str, object] = {
            "dimension": int(simple_index.dimension),
            "vectors": vectors_payload,
        }
        with Path(path).open("wb") as handle:
            safe_pickle_dump(payload, handle)

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

        Raises
        ------
        AgentCatalogSearchError
            If the persisted index has an invalid payload format.
        """
        with Path(path).open("rb") as handle:
            # Load from trusted local artifact with allow-list validation
            # (see orchestration.safe_pickle for type restrictions)
            payload_raw: object = safe_pickle_load(handle)

        # Strict validation: reject if not a dict or has unexpected keys
        if not isinstance(payload_raw, dict):
            message = "Stored semantic index has invalid payload format"
            raise AgentCatalogSearchError(message)

        # Type-narrowed payload for subsequent operations
        payload: dict[str, object] = cast(dict[str, object], payload_raw)
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
    def normalize_l2(vectors: VectorArray) -> None:
        """Normalize vectors to unit length in-place.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Array to normalize (modified in-place).
        """
        normalized = _normalize_l2_array(np.asarray(vectors, dtype=np.float32, order="C"), axis=1)
        np.copyto(vectors, normalized)


def _with_cache(func: Callable[[], FaissModuleProtocol]) -> Callable[[], FaissModuleProtocol]:
    """Cache a parameterless factory function.

    Wraps functools.cache with proper type annotations to avoid mypy issues
    with incomplete functools stubs.

    Parameters
    ----------
    func : Callable[[], FaissModuleProtocol]
        Zero-argument factory function.

    Returns
    -------
    Callable[[], FaissModuleProtocol]
        Cached version of the factory function.
    """
    return cache(func)


@_with_cache
def _simple_faiss_module() -> FaissModuleProtocol:
    """Return a cached NumPy-based FAISS shim for local usage.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    FaissModuleProtocol
        Describe return value.
    """
    return _simple_faiss_module()


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
        has_index_builder = hasattr(module, "IndexFlatIP") or hasattr(module, "index_flat_ip")
        has_id_map = hasattr(module, "IndexIDMap2") or hasattr(module, "index_id_map2")
        has_writer = hasattr(module, "write_index")
        if has_index_builder and has_id_map and has_writer:
            adapted = wrap_faiss_module(module)
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
            return adapted
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
    return wrap_faiss_module(_SimpleFaissModule())


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
    tokens: list[str] = WORD_PATTERN.findall(text.lower())
    return tokens


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


def _extract_agent_hints_payload(symbol: CatalogMapping) -> tuple[list[str], list[str]]:
    """Return curated ``intent_tags`` and ``tests_to_run`` lists for ``symbol``."""
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
    docfacts: Mapping[str, JsonLike] | None,
) -> tuple[str | None, str | None]:
    """Return summary/docstring text pulled from the ``docfacts`` mapping."""
    summary = None
    docstring = None
    if isinstance(docfacts, Mapping):
        summary_val = docfacts.get("summary")
        docstring_val = docfacts.get("docstring")
        summary = _stringify(summary_val)
        docstring = _stringify(docstring_val)
    return summary, docstring


def _extract_anchor_lines(
    symbol: CatalogMapping,
) -> tuple[int | None, int | None]:
    """Return source anchor line numbers for ``symbol`` when present."""
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
    symbol: CatalogMapping,
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
    # Type-narrow docfacts_payload: only pass if it's a Mapping
    docfacts_input: Mapping[str, JsonLike] | None = (
        cast(Mapping[str, JsonLike], docfacts_payload)
        if isinstance(docfacts_payload, Mapping)
        else None
    )
    summary, docstring = _extract_docfacts_text(docfacts_input)
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
    catalog: CatalogMapping,
) -> Sequence[tuple[str, str, CatalogMapping]]:
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
            CatalogMapping,
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
            module_entries = [
                (package_name, module_name, symbol)
                for module in modules
                if isinstance(module, Mapping)
                for module_name, symbols in [
                    (_stringify(module.get("qualified")), module.get("symbols"))
                ]
                if module_name and isinstance(symbols, list)
                for symbol in symbols
                if isinstance(symbol, Mapping)
            ]
            entries.extend(module_entries)
    return entries


def documents_from_catalog(
    catalog: CatalogMapping,
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
    search_config: CatalogMapping | None,
    options: SearchOptions,
    document_count: int,
    k: int,
) -> tuple[float, int]:
    """Return the `(alpha_value, candidate_limit)` pair for catalog search."""
    config = search_config or {}

    alpha_value = options.alpha
    if alpha_value is None:
        alpha_candidate = config.get("alpha")
        if isinstance(alpha_candidate, (int, float)):
            alpha_value = float(alpha_candidate)
    if alpha_value is None:
        alpha_value = 0.6
    alpha_value = min(max(alpha_value, 0.0), 1.0)

    candidate_value = options.candidate_pool
    if candidate_value is None:
        pool_value = config.get("candidate_pool")
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
    catalog: CatalogMapping,
    repo_root: Path,
) -> tuple[CatalogMapping, Path, Path] | None:
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
    semantic_meta_typed: CatalogMapping = cast(
        CatalogMapping,
        semantic_meta,
    )
    return semantic_meta_typed, index_path, mapping_path


def _load_row_lookup(
    mapping_path: Path,
) -> tuple[dict[str, int], CatalogMapping]:
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
    # Load JSON and immediately validate the top-level type
    mapping_payload_raw: object = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping_payload_raw, dict):
        message = "Semantic mapping file does not contain valid JSON object"
        raise AgentCatalogSearchError(message)
    # Type-narrowed payload for subsequent operations
    mapping_payload: CatalogMapping = cast(
        CatalogMapping,
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
    # getattr with default None returns Any; we immediately narrow with callable check
    factory_raw: object = getattr(module, "SentenceTransformer", None)
    factory: Callable[[str], object] | None = factory_raw if callable(factory_raw) else None
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
    options: SearchOptions, semantic_meta: PrimitiveMapping
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
) -> FloatMatrix:
    """Return a single-row embedding matrix for ``query``."""
    sentences = [query]
    embeddings = model.encode(sentences, batch_size=batch_size)
    return np.asarray(embeddings, dtype=np.float32, order="C")


def _extract_semantic_metadata(raw_meta: CatalogMapping) -> PrimitiveMapping:
    return cast(
        PrimitiveMapping,
        {
            key: value
            for key, value in raw_meta.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    )


def _load_faiss_index(index_path: Path) -> FaissIndexProtocol:
    module = load_faiss("vector search")
    return module.read_index(str(index_path))


def _prepare_vector_search_inputs(
    options: SearchOptions,
    context: VectorSearchContext,
) -> _VectorSearchInputs:
    semantic_meta = _extract_semantic_metadata(context.semantic_meta)
    _, model = _resolve_embedding_model(options, semantic_meta)
    batch_size = options.batch_size or _DEFAULT_BATCH_SIZE
    index = _load_faiss_index(context.index_path)
    return _VectorSearchInputs(
        model=model,
        batch_size=batch_size,
        index=index,
        candidate_limit=context.candidate_limit,
        row_lookup=context.row_to_document,
    )


def _scores_from_indices(
    distances: FloatMatrix,
    indices: IntVector,
    row_lookup: Mapping[int, SearchDocument],
) -> dict[str, float]:
    if distances.size == 0 or indices.size == 0:
        return {}

    query_rows = np.asarray(indices[0, :], dtype=np.int64, order="C")
    score_rows = np.asarray(distances[0, :], dtype=np.float32, order="C")
    row_sequence = cast(Sequence[int], query_rows.tolist())
    score_sequence = cast(Sequence[float], score_rows.tolist())
    row_list: list[int] = [int(value) for value in row_sequence]
    score_list: list[float] = [float(value) for value in score_sequence]
    scores: dict[str, float] = {}
    for row_id, score in zip(row_list, score_list, strict=False):
        if row_id < 0:
            continue
        document = row_lookup.get(row_id)
        if document is not None:
            scores[document.symbol_id] = float(score)
    return scores


def compute_vector_scores(
    query: str,
    options: SearchOptions,
    context: VectorSearchContext,
) -> dict[str, float]:
    """Compute vector similarity scores for candidates using semantic index.

    Parameters
    ----------
    query : str
        Search query text to encode.
    options : SearchOptions
        Search configuration including embedding model and batch size.
    context : VectorSearchContext
        Pre-loaded vector search context with index and documents.

    Returns
    -------
    dict[str, float]
        Mapping from symbol_id to vector similarity score.

    Raises
    ------
    AgentCatalogSearchError
        If vector encoding or search fails.
    """
    inputs = _prepare_vector_search_inputs(options, context)
    query_embedding = _encode_query(inputs.model, query, batch_size=inputs.batch_size)
    query_normalized = _normalize_l2_array(query_embedding, axis=1)
    distances, indices = inputs.index.search(query_normalized, inputs.candidate_limit)
    return _scores_from_indices(distances, indices, inputs.row_lookup)


def _build_vector_search_context(
    catalog: Mapping[str, JsonLike],
    request: SearchRequest,
    lexical_candidates: Sequence[SearchDocument],
    candidate_limit: int,
    documents: Sequence[SearchDocument],
) -> VectorSearchContext | None:
    semantic_index_meta = _resolve_semantic_index_metadata(catalog, request.repo_root)
    if semantic_index_meta is None:
        return None

    semantic_meta, index_path, mapping_path = semantic_index_meta
    _, mapping_payload = _load_row_lookup(mapping_path)

    return VectorSearchContext(
        semantic_meta=semantic_meta,
        mapping_payload=mapping_payload,
        index_path=index_path,
        documents=lexical_candidates,
        candidate_limit=candidate_limit,
        k=request.k,
        candidate_ids={doc.symbol_id for doc in lexical_candidates},
        row_to_document={doc.row: doc for doc in documents if doc.row >= 0},
    )


def _compute_vector_scores_safe(
    query: str,
    options: SearchOptions,
    context: VectorSearchContext | None,
) -> dict[str, float]:
    if context is None:
        return {}
    try:
        return compute_vector_scores(query, options, context)
    except AgentCatalogSearchError:
        return {}


def search_catalog(
    catalog: Mapping[str, JsonLike],
    *,
    request: SearchRequest,
    options: SearchOptions | None = None,
    metrics: MetricsProvider | None = None,
) -> list[SearchResult]:
    """Execute hybrid (lexical + vector) search across the agent catalog.

    Combines lexical term matching with semantic vector similarity to rank
    catalog entries. Uses typed NumPy helpers to ensure shape safety and
    predictable results.

    Parameters
    ----------
    catalog : Mapping[str, JsonLike]
        Agent catalog payload (from agent_catalog.json).
    request : SearchRequest
        Search parameters including query, k (result count), and repo_root.
    options : SearchOptions, optional
        Tuning parameters including facets, alpha weighting, model selection.
    metrics : MetricsProvider, optional
        Optional observability provider for metrics/logging.

    Returns
    -------
    list[SearchResult]
        Sorted list of top-k search results with combined scores.

    Raises
    ------
    AgentCatalogSearchError
        If catalog parsing, indexing, or search fails.

    Examples
    --------
    >>> from pathlib import Path
    >>> from kgfoundry.agent_catalog.search import SearchRequest, search_catalog
    >>> # Typically used via AgentCatalogClient
    >>> # This is a conceptual example - real usage requires a valid catalog
    """
    if metrics is None:
        metrics = MetricsProvider.default()

    opts = options or SearchOptions()
    documents = documents_from_catalog(catalog)
    lexical_scores = compute_lexical_scores(
        prepare_query_tokens(request.query), documents, request.query
    )

    search_config = catalog.get("search")
    alpha_value, candidate_limit = resolve_search_parameters(
        cast(
            CatalogMapping | None,
            search_config if isinstance(search_config, Mapping) else None,
        ),
        opts,
        len(documents),
        request.k,
    )

    lexical_candidates = select_lexical_candidates(lexical_scores, documents, candidate_limit)
    vector_context = _build_vector_search_context(
        catalog,
        request,
        lexical_candidates,
        candidate_limit,
        documents,
    )
    vector_scores = _compute_vector_scores_safe(request.query, opts, vector_context)

    results: list[SearchResult] = []
    facets = opts.facets or {}
    for doc in lexical_candidates:
        if facets and not document_matches_facets(doc, facets):
            continue

        lexical_score = lexical_scores.get(doc.symbol_id, 0.0)
        vector_score = vector_scores.get(doc.symbol_id, 0.0)
        results.append(
            SearchResult(
                symbol_id=doc.symbol_id,
                score=alpha_value * vector_score + (1.0 - alpha_value) * lexical_score,
                lexical_score=lexical_score,
                vector_score=vector_score,
                package=doc.package,
                module=doc.module,
                qname=doc.qname,
                kind=doc.kind,
                stability=doc.stability,
                deprecated=doc.deprecated,
                summary=doc.summary,
                docstring=doc.docstring,
                anchor={"start_line": doc.anchor_start, "end_line": doc.anchor_end},
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: request.k]
