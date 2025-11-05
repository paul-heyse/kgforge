"""Hybrid search utilities for the agent catalog.

This module orchestrates lexical and vector search across the catalog payload.
Vector operations rely on the typed helpers in
``kgfoundry_common.numpy_typing`` so that downstream consumers (and static type checkers) can
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, cast

import numpy as np

from kgfoundry_common.errors import AgentCatalogSearchError
from kgfoundry_common.logging import get_logger
from kgfoundry_common.numpy_typing import (
    normalize_l2 as _normalize_l2_array,
)
from kgfoundry_common.numpy_typing import (
    topk_indices,
)
from kgfoundry_common.observability import MetricsProvider
from orchestration.safe_pickle import dump as safe_pickle_dump
from orchestration.safe_pickle import load as safe_pickle_load
from search_api.types import (
    FaissIndexProtocol,
    wrap_faiss_module,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kgfoundry_common.numpy_typing import (
        FloatMatrix,
        FloatVector,
        IntVector,
    )
    from search_api.types import (
        FaissModuleProtocol,
        IndexArray,
        VectorArray,
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

    This TypedDict defines the complete search options with all optional fields. It aligns with JSON
    Schema definitions and ensures type safety.
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

    This TypedDict ensures all required fields are present and properly typed, providing parity with
    the JSON Schema definition for search documents.
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
    """Optional overrides for SearchDocument construction.

    All fields are optional and can be used to override default values
    when creating SearchDocument instances via helper functions.

    Parameters
    ----------
    stability : str | None, optional
        Stability level (e.g., "stable", "experimental").
    deprecated : bool, optional
        Whether the symbol is deprecated.
    summary : str | None, optional
        Short summary text extracted from docstring.
    docstring : str | None, optional
        Full docstring text.
    anchor_start : int | None, optional
        Starting line number for source anchor.
    anchor_end : int | None, optional
        Ending line number for source anchor.
    row : int, optional
        Row index in semantic index mapping.
    """

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

    Defines the interface that embedding models must implement for use with the catalog search
    system. Models must provide an encode method that takes sentences and returns vector arrays.
    """

    def encode(self, sentences: Sequence[str], **_: object) -> VectorArray:
        """Return embeddings for the provided sentences.

        Encodes a sequence of sentences into a vector array. Each sentence
        becomes a row in the resulting matrix.

        Parameters
        ----------
        sentences : Sequence[str]
            Sequence of sentences to encode.
        **_ : object
            Additional keyword arguments (unused, for extensibility).

        Returns
        -------
        VectorArray
            Vector array with shape (n_sentences, embedding_dim).
        """
        ...


@dataclass(slots=True)
class SearchConfig:
    """Configuration used for hybrid search against the catalog.

    Configuration parameters for hybrid search combining lexical and
    vector search. Controls the balance between search methods and
    candidate pool size.

    Parameters
    ----------
    alpha : float
        Hybrid search weighting factor (0.0 = lexical only, 1.0 = vector only).
    candidate_pool : int
        Number of candidates to retrieve before final ranking.
    lexical_fields : list[str]
        List of document fields to use for lexical search.
    """

    alpha: float
    candidate_pool: int
    lexical_fields: list[str]


@dataclass(slots=True)
class SearchOptions:
    """Optional tuning parameters for hybrid search.

    Optional configuration parameters for fine-tuning hybrid search
    behavior. All fields are optional and have sensible defaults when
    not provided.

    Parameters
    ----------
    alpha : float | None, optional
        Hybrid search weighting factor (0.0 = lexical, 1.0 = vector).
        Defaults to None (uses default 0.6).
    facets : Mapping[str, str] | None, optional
        Facet filters for pre-filtering results (package, module, kind, stability).
        Defaults to None.
    candidate_pool : int | None, optional
        Pre-filtering candidate pool size. Must be >= k.
        Defaults to None (uses default 100).
    model_loader : Callable[[str], EmbeddingModelProtocol] | None, optional
        Factory function to load embedding models dynamically.
        Defaults to None.
    embedding_model : str | None, optional
        Name or path of embedding model to use.
        Defaults to None.
    batch_size : int | None, optional
        Batch size for embedding model encoding.
        Defaults to None (uses default 32).
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

    Notes
    -----
    Propagates :class:`AgentCatalogSearchError` when facet keys are not in the
    allow-list.
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

    Represents a searchable document in the catalog with all metadata
    needed for hybrid search. Includes symbol information, text content,
    tokens for lexical search, and optional anchor positions.

    Parameters
    ----------
    symbol_id : str
        Unique symbol identifier (e.g., "py:kgfoundry.search.find_similar").
    package : str
        Package name containing the symbol.
    module : str
        Fully qualified module name.
    qname : str
        Fully qualified symbol name.
    kind : str
        Symbol kind (e.g., "class", "function", "module").
    stability : str | None
        Stability level (e.g., "stable", "experimental", "deprecated").
    deprecated : bool
        Whether the symbol is deprecated.
    summary : str | None
        Short summary text extracted from docstring.
    docstring : str | None
        Full docstring text.
    anchor_start : int | None
        Starting line number for source anchor.
    anchor_end : int | None
        Ending line number for source anchor.
    text : str
        Normalized text content for lexical search.
    tokens : collections.Counter[str]
        Token frequency counter for lexical scoring.
    row : int, optional
        Row index in semantic index mapping. Defaults to -1.
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

    Represents a single search result with symbol metadata and computed
    scores. Includes both lexical and vector scores along with the final
    hybrid score.

    Parameters
    ----------
    symbol_id : str
        Unique symbol identifier.
    score : float
        Final hybrid search score (weighted combination of lexical and vector).
    lexical_score : float
        Lexical search score component.
    vector_score : float
        Vector search score component.
    package : str
        Package name containing the symbol.
    module : str
        Fully qualified module name.
    qname : str
        Fully qualified symbol name.
    kind : str
        Symbol kind (e.g., "class", "function", "module").
    stability : str | None
        Stability level (e.g., "stable", "experimental").
    deprecated : bool
        Whether the symbol is deprecated.
    summary : str | None
        Short summary text.
    anchor_start : int | None
        Starting line number for source anchor.
    anchor_end : int | None
        Ending line number for source anchor.
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

    Provides a NumPy-based fallback implementation of FaissIndexProtocol
    for environments where FAISS is not installed. Uses cosine similarity
    via matrix multiplication for vector search.

    Parameters
    ----------
    dimension : int
        Vector dimension for the index.
    """

    def __init__(self, dimension: int) -> None:
        """Initialize simple FAISS index.

        Creates a new simple index with the specified vector dimension.
        Starts with an empty vector store.

        Parameters
        ----------
        dimension : int
            Vector dimension for the index.
        """
        self.dimension = dimension
        self._vectors: FloatMatrix = np.empty((0, dimension), dtype=np.float32, order="C")

    def add(self, vectors: VectorArray) -> None:
        """Add vectors to the index.

        Adds vectors to the index, validating dimension compatibility.
        Vectors are stored in a contiguous float32 matrix.

        Parameters
        ----------
        vectors : VectorArray
            Vectors to add (shape: (n_vectors, dimension)).

        Raises
        ------
        AgentCatalogSearchError
            If vector dimension does not match index configuration.
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

    @property
    def vectors(self) -> FloatMatrix:
        """Return the stored vectors for serialization or inspection."""
        return self._vectors

    def load_vectors(self, vectors: FloatMatrix) -> None:
        """Replace the stored vectors with ``vectors`` in contiguous form."""
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def train(self, vectors: VectorArray) -> None:
        """Train the index (no-op for simple flat index).

        Simple flat index doesn't require training, so this is a no-op.
        Provided for protocol compatibility.

        Parameters
        ----------
        vectors : VectorArray
            Training vectors (unused for simple index).
        """
        # Simple flat index doesn't require training

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:
        """Add vectors with explicit IDs (not supported by simple index).

        Validates that IDs length matches vector count, then falls back
        to regular add() since simple index doesn't support ID mapping.

        Parameters
        ----------
        vectors : VectorArray
            Vectors to add (shape: (n_vectors, dimension)).
        ids : IndexArray
            Vector IDs (shape: (n_vectors,)).

        Raises
        ------
        AgentCatalogSearchError
            If IDs length does not match vector count.
        """
        id_array = np.asarray(ids, dtype=np.int64)
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.shape[0] != id_array.shape[0]:
            message = "IDs length must match vector count"
            raise AgentCatalogSearchError(message)
        # Simple index doesn't support ID mapping, fall back to regular add
        self.add(vector_array)

    def search(self, vectors: VectorArray, k: int) -> tuple[FloatMatrix, IntVector]:
        """Search for k nearest neighbors.

        Performs cosine similarity search using matrix multiplication.
        Returns distances and indices for the top-k results per query.

        Parameters
        ----------
        vectors : VectorArray
            Query vectors (shape: (n_queries, dimension)).
        k : int
            Number of neighbors to retrieve per query.

        Returns
        -------
        tuple[FloatMatrix, IntVector]
            Tuple of (distances, indices) where distances has shape
            (n_queries, k) and indices has shape (n_queries, k).

        Raises
        ------
        AgentCatalogSearchError
            If query vector dimension does not match index configuration.
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
        similarity_matrix: FloatMatrix = cast("FloatMatrix", queries @ self._vectors.T)
        if similarity_matrix.ndim != EMBEDDING_MATRIX_RANK:
            message = "Unexpected similarity matrix shape"
            raise AgentCatalogSearchError(message)
        top_k = min(k, similarity_matrix.shape[1])
        for row_idx in range(similarity_matrix.shape[0]):
            scores_row = cast("FloatVector", similarity_matrix[row_idx])
            top_indices = topk_indices(scores_row, top_k)
            distances[row_idx, :top_k] = scores_row[top_indices]
            indices[row_idx, :top_k] = top_indices
        return distances, indices


class _SimpleFaissModule:
    """Minimal FAISS module shim using NumPy for tests and local runs.

    Implements FaissModuleProtocol for compatibility with FAISS adapters. Provides a simple NumPy-
    based fallback when FAISS is not available.
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

        Creates a simple flat index optimized for inner-product similarity.
        Uses NumPy-based implementation.

        Parameters
        ----------
        dimension : int
            Vector dimension for the index.

        Returns
        -------
        FaissIndexProtocol
            Flat index instance.
        """
        return cast("FaissIndexProtocol", _SimpleFaissModule._create_flat_index(dimension))

    @staticmethod
    def index_factory(dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        """Create an index from a factory string.

        For the simple implementation, factory strings are ignored and a flat
        index is always returned. Provided for protocol compatibility.

        Parameters
        ----------
        dimension : int
            Vector dimension.
        factory_string : str
            Factory description (ignored in simple implementation).
        metric : int
            Metric type (METRIC_INNER_PRODUCT or METRIC_L2) (ignored in simple
            implementation).

        Returns
        -------
        FaissIndexProtocol
            Flat index instance.
        """
        del factory_string, metric
        # Simple implementation ignores factory configuration and always returns flat index
        return cast("FaissIndexProtocol", _SimpleFaissModule._create_flat_index(dimension))

    @staticmethod
    def index_id_map2(index: FaissIndexProtocol) -> FaissIndexProtocol:
        """Wrap an index with 64-bit ID mapping.

        For the simple implementation, this is a no-op (returns the index as-is).
        Provided for protocol compatibility.

        Parameters
        ----------
        index : FaissIndexProtocol
            Base index to wrap.

        Returns
        -------
        FaissIndexProtocol
            The same index instance (no wrapping in simple implementation).
        """
        return cast("FaissIndexProtocol", _SimpleFaissModule._ensure_simple_index(index))

    @staticmethod
    def write_index(index: FaissIndexProtocol, path: str) -> None:
        """Persist an index to disk.

        Saves a simple FAISS index to disk using safe pickle serialization.
        Stores dimension and vectors in a dictionary payload.

        Parameters
        ----------
        index : FaissIndexProtocol
            Index instance to save.
        path : str
            File path for the persisted index.
        """
        simple_index = _SimpleFaissModule._ensure_simple_index(index)
        vectors_payload = cast("list[list[float]]", simple_index.vectors.tolist())
        payload: dict[str, object] = {
            "dimension": int(simple_index.dimension),
            "vectors": vectors_payload,
        }
        with Path(path).open("wb") as handle:
            safe_pickle_dump(payload, handle)

    @staticmethod
    def read_index(path: str) -> FaissIndexProtocol:
        """Load an index from disk.

        Loads a simple FAISS index from disk using safe pickle deserialization.
        Validates payload format and reconstructs the index.

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
            If the persisted index has an invalid payload format or dimension.
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
        payload: dict[str, object] = cast("dict[str, object]", payload_raw)
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
            fallback.load_vectors(vectors)
        return cast("FaissIndexProtocol", fallback)

    @staticmethod
    def normalize_l2(vectors: VectorArray) -> None:
        """Normalize vectors to unit length in-place.

        Normalizes vectors along axis 1 using L2 normalization.
        Modifies the input array in-place.

        Parameters
        ----------
        vectors : VectorArray
            Array to normalize (modified in-place, shape: (n_vectors, dimension)).
        """
        normalized = _normalize_l2_array(np.asarray(vectors, dtype=np.float32, order="C"), axis=1)
        np.copyto(vectors, normalized)


def _with_cache(func: Callable[[], FaissModuleProtocol]) -> Callable[[], FaissModuleProtocol]:
    """Cache a parameterless factory function.

    Wraps functools.cache with proper type annotations to avoid type checker issues
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

    Returns a cached instance of the simple FAISS module shim.
    Uses functools.cache to ensure only one instance is created.

    Returns
    -------
    FaissModuleProtocol
        Cached simple FAISS module instance.
    """
    return _simple_faiss_module()


def load_faiss(purpose: str) -> FaissModuleProtocol:
    """Import a FAISS module or fall back to the NumPy implementation.

    Attempts to import FAISS from configured modules, falling back to
    NumPy-based implementation if unavailable. Respects environment variables
    for module override and fallback control.

    Parameters
    ----------
    purpose : str
        Purpose description for logging (e.g., "search", "index_build").

    Returns
    -------
    FaissModuleProtocol
        FAISS module protocol instance (real FAISS or simple fallback).

    Raises
    ------
    AgentCatalogSearchError
        If FAISS is required but cannot be imported and fallback is disabled.
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
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
        ) as exc:  # pragma: no cover - runtime guard
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

    Extracts alphanumeric tokens from text using WORD_PATTERN regex.
    Converts text to lowercase before tokenization.

    Parameters
    ----------
    text : str
        Input text to tokenize.

    Returns
    -------
    list[str]
        List of normalized word tokens.
    """
    tokens: list[str] = WORD_PATTERN.findall(text.lower())
    return tokens


def _stringify(value: object) -> str | None:
    """Return ``value`` as ``str`` when it is not ``None``.

    Converts a value to string if it's not None, otherwise returns None.
    Used for safely extracting string values from catalog mappings.

    Parameters
    ----------
    value : object
        Value to convert to string.

    Returns
    -------
    str | None
        String representation of value, or None if value is None.
    """
    if value is None:
        return None
    return str(value)


def _extract_agent_hints_payload(symbol: CatalogMapping) -> tuple[list[str], list[str]]:
    """Return curated ``intent_tags`` and ``tests_to_run`` lists for ``symbol``.

    Parameters
    ----------
    symbol : CatalogMapping
        Symbol mapping to extract hints from.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple of (intent_tags, tests_to_run) lists.
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
    docfacts: Mapping[str, JsonLike] | None,
) -> tuple[str | None, str | None]:
    """Return summary/docstring text pulled from the ``docfacts`` mapping.

    Parameters
    ----------
    docfacts : Mapping[str, JsonLike] | None
        Docfacts mapping to extract text from.

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of (summary, docstring) strings.
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
    symbol: CatalogMapping,
) -> tuple[int | None, int | None]:
    """Return source anchor line numbers for ``symbol`` when present.

    Parameters
    ----------
    symbol : CatalogMapping
        Symbol mapping to extract anchor lines from.

    Returns
    -------
    tuple[int | None, int | None]
        Tuple of (start_line, end_line) numbers.
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
    symbol: CatalogMapping,
    symbol_id: str,
    row: int,
) -> SearchDocument:
    """Create a ``SearchDocument`` from the raw symbol payload.

    Extracts all relevant fields from a catalog symbol mapping and constructs
    a SearchDocument with normalized text and tokenization. Handles docfacts,
    agent hints, anchor lines, and metrics.

    Parameters
    ----------
    package_name : str
        Package name containing the symbol.
    module_name : str
        Fully qualified module name.
    symbol : CatalogMapping
        Symbol mapping from catalog payload.
    symbol_id : str
        Unique symbol identifier.
    row : int
        Row index in semantic index mapping.

    Returns
    -------
    SearchDocument
        Constructed search document with all fields populated.
    """
    docfacts_payload = symbol.get("docfacts")
    # Type-narrow docfacts_payload: only pass if it's a Mapping
    docfacts_input: Mapping[str, JsonLike] | None = (
        cast("Mapping[str, JsonLike]", docfacts_payload)
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

    Iterates through the catalog structure to extract all symbols along with
    their package and module names. Returns a sequence of triples.

    Parameters
    ----------
    catalog : CatalogMapping
        Catalog payload mapping containing packages, modules, and symbols.

    Returns
    -------
    Sequence[tuple[str, str, CatalogMapping]]
        Sequence of (package_name, module_name, symbol) triples.
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

    Builds SearchDocument instances from all symbols in the catalog.
    Optionally maps symbol IDs to row indices if row_lookup is provided.

    Parameters
    ----------
    catalog : CatalogMapping
        Catalog payload mapping containing packages, modules, and symbols.
    row_lookup : Mapping[str, int] | None, optional
        Optional mapping from symbol_id to row index. If provided, symbols
        without entries are skipped. Defaults to None.

    Returns
    -------
    list[SearchDocument]
        List of search documents constructed from catalog symbols.
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

    Parses common string representations of boolean values.
    Returns None if the value cannot be interpreted as a boolean.

    Parameters
    ----------
    value : str
        String value to parse (e.g., "true", "1", "yes", "false", "0", "no").

    Returns
    -------
    bool | None
        True for "1", "true", "yes", "on"; False for "0", "false", "no", "off";
        None for unrecognized values.
    """
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def document_matches_facets(document: SearchDocument, facets: Mapping[str, str]) -> bool:
    """Return ``True`` when ``document`` satisfies the provided facets.

    Checks if a document matches all provided facet filters. Supported facet
    keys are: package, module, kind, and stability.

    Parameters
    ----------
    document : SearchDocument
        Document to check against facets.
    facets : Mapping[str, str]
        Facet filters to apply (package, module, kind, stability).

    Returns
    -------
    bool
        True if document matches all facets, False otherwise.
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

    Tokenizes the query using WORD_PATTERN regex. Falls back to simple
    whitespace splitting if no tokens are found.

    Parameters
    ----------
    query : str
        Search query text.

    Returns
    -------
    collections.Counter[str]
        Token frequency counter for the query.
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
    """Return the `(alpha_value, candidate_limit)` pair for catalog search.

    Parameters
    ----------
    search_config : CatalogMapping | None
        Search configuration mapping.
    options : SearchOptions
        Search options instance.
    document_count : int
        Total number of documents in the catalog.
    k : int
        Number of results requested.

    Returns
    -------
    tuple[float, int]
        Tuple of (alpha_value, candidate_limit).
    """
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

    Computes lexical similarity scores by matching query tokens against
    document tokens. Provides a boost if the query appears in the document's
    qualified name.

    Parameters
    ----------
    query_tokens : collections.Counter[str]
        Token frequency counter for the query.
    documents : Sequence[SearchDocument]
        Candidate documents to score.
    query : str
        Original query text (for qname matching).

    Returns
    -------
    dict[str, float]
        Mapping from symbol_id to lexical score.
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

    Sorts documents by lexical score and returns the top candidates.

    Parameters
    ----------
    lexical_scores : Mapping[str, float]
        Mapping from symbol_id to lexical score.
    documents : Sequence[SearchDocument]
        Candidate documents to filter.
    candidate_limit : int
        Maximum number of candidates to return.

    Returns
    -------
    list[SearchDocument]
        Top-scoring documents sorted by lexical score (descending).
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

    Extracts semantic index metadata from catalog and validates that
    artifact paths exist and are under repo_root (prevents directory traversal).

    Parameters
    ----------
    catalog : CatalogMapping
        Catalog payload mapping containing semantic_index metadata.
    repo_root : Path
        Repository root for resolving relative artifact paths.

    Returns
    -------
    tuple[CatalogMapping, Path, Path] | None
        Tuple of (semantic_meta, index_path, mapping_path) if available,
        None if semantic_index is not present.

    Raises
    ------
    AgentCatalogSearchError
        If semantic index metadata is missing, artifact paths are invalid,
        or artifacts are missing from disk.
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
        "CatalogMapping",
        semantic_meta,
    )
    return semantic_meta_typed, index_path, mapping_path


def _load_row_lookup(
    mapping_path: Path,
) -> tuple[dict[str, int], CatalogMapping]:
    """Return the row lookup mapping and raw payload from ``mapping_path``.

    Loads the semantic mapping JSON file and extracts symbol_id to row index
    mappings. Validates JSON structure and returns both the lookup dict and
    the raw payload.

    Parameters
    ----------
    mapping_path : Path
        Path to semantic mapping JSON file.

    Returns
    -------
    tuple[dict[str, int], CatalogMapping]
        Tuple of (row_lookup dictionary, mapping_payload).

    Raises
    ------
    AgentCatalogSearchError
        If mapping file does not contain valid JSON object or symbols list.
    """
    # Load JSON and immediately validate the top-level type
    mapping_payload_raw: object = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping_payload_raw, dict):
        message = "Semantic mapping file does not contain valid JSON object"
        raise AgentCatalogSearchError(message)
    # Type-narrowed payload for subsequent operations
    mapping_payload: CatalogMapping = cast(
        "CatalogMapping",
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

    Dynamically imports sentence-transformers and loads the specified model.
    Provides clear error messages if the library is unavailable or model loading fails.

    Parameters
    ----------
    model_name : str
        Name or path of the sentence-transformers model to load.

    Returns
    -------
    EmbeddingModelProtocol
        Loaded embedding model instance.

    Raises
    ------
    AgentCatalogSearchError
        If sentence-transformers cannot be imported, SentenceTransformer class
        is missing, or model cannot be loaded.
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
    return cast("EmbeddingModelProtocol", model)


def _resolve_embedding_model(
    options: SearchOptions, semantic_meta: PrimitiveMapping
) -> tuple[str, EmbeddingModelProtocol]:
    """Return the embedding model name and instance used for vector search.

    Resolves the embedding model from options or semantic metadata.
    Uses the model loader from options or defaults to sentence-transformers.

    Parameters
    ----------
    options : SearchOptions
        Search options containing optional model name and loader.
    semantic_meta : PrimitiveMapping
        Semantic metadata mapping containing optional model name.

    Returns
    -------
    tuple[str, EmbeddingModelProtocol]
        Tuple of (model_name, model_instance).

    Raises
    ------
    AgentCatalogSearchError
        If semantic index metadata does not include model name or model
        cannot be loaded.
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
    """Return a single-row embedding matrix for ``query``.

    Parameters
    ----------
    model : EmbeddingModelProtocol
        Embedding model to use for encoding.
    query : str
        Query string to encode.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    FloatMatrix
        Single-row embedding matrix.
    """
    sentences = [query]
    embeddings = model.encode(sentences, batch_size=batch_size)
    return np.asarray(embeddings, dtype=np.float32, order="C")


def _extract_semantic_metadata(raw_meta: CatalogMapping) -> PrimitiveMapping:
    return cast(
        "PrimitiveMapping",
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
    row_sequence = cast("Sequence[int]", query_rows.tolist())
    score_sequence = cast("Sequence[float]", score_rows.tolist())
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

    Notes
    -----
    Propagates :class:`AgentCatalogSearchError` when vector encoding or search
    fails.
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
    """Build vector search context from catalog metadata.

    Parameters
    ----------
    catalog : Mapping[str, JsonLike]
        Catalog payload.
    request : SearchRequest
        Search request parameters.
    lexical_candidates : Sequence[SearchDocument]
        Lexical search candidates.
    candidate_limit : int
        Maximum number of candidates.
    documents : Sequence[SearchDocument]
        All documents in catalog.

    Returns
    -------
    VectorSearchContext | None
        Vector search context if semantic index is available, None otherwise.

    Notes
    -----
    Propagates :class:`AgentCatalogSearchError` when semantic index metadata is
    invalid or artifacts are missing.
    """
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

    Notes
    -----
    Propagates :class:`AgentCatalogSearchError` when catalog parsing, indexing,
    or search fails.

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
            "CatalogMapping | None",
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
