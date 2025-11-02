"""Vector search protocols and typed result structures.

This module defines PEP 544 protocols for FAISS-compatible vector search
and typed result structures using TypedDict/dataclass. All protocols and
types are fully typed to enable mypy strict checking.

Examples
--------
>>> from kgfoundry.search_api.types import VectorSearchResult, FaissIndexProtocol
>>> from numpy.typing import NDArray
>>> import numpy as np
>>> # Protocol usage
>>> class MyIndex:
...     def add(self, vectors: NDArray[np.float32]) -> None: ...
...     def search(
...         self, vectors: NDArray[np.float32], k: int
...     ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
>>> # Verify protocol satisfaction
>>> def use_index(idx: FaissIndexProtocol) -> None:
...     vecs = np.array([[1.0, 2.0]], dtype=np.float32)
...     idx.add(vecs)
...     distances, indices = idx.search(vecs, k=5)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import JsonValue

__all__ = [
    "AgentSearchQuery",
    "AgentSearchResponse",
    "BM25IndexProtocol",
    "FaissIndexProtocol",
    "FaissModuleProtocol",
    "GpuClonerOptionsProtocol",
    "GpuResourcesProtocol",
    "IndexArray",
    "SpladeEncoderProtocol",
    "VectorArray",
    "VectorSearchResult",
    "VectorSearchResultTypedDict",
]

__navmap__: Final[NavMap] = {
    "title": "search_api.types",
    "synopsis": "Vector search protocols and typed result structures",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}

# Type aliases for numpy arrays
if TYPE_CHECKING:
    import numpy.typing as npt

    type VectorArray = npt.NDArray[np.float32]
    type IndexArray = npt.NDArray[np.int64]
else:  # pragma: no cover - runtime fallback for doc generation
    VectorArray = np.ndarray
    IndexArray = np.ndarray

VectorArray.__doc__ = (
    "Type alias for float32 vector arrays used in FAISS operations.\n\n"
    "All vectors must be normalized to unit length for inner-product search.\n"
    "Dimensions are typically 2560 for dense embeddings or configurable for\n"
    "sparse representations."
)

IndexArray.__doc__ = (
    "Type alias for int64 index arrays used in FAISS search results.\n\n"
    "Index arrays contain row indices into the vector store, typically returned\n"
    "from search operations alongside distance/similarity scores."
)


class FaissIndexProtocol(Protocol):
    """Protocol for FAISS-compatible vector indexes.

    <!-- auto:docstring-builder v1 -->

    This protocol defines the minimum interface required for vector search
    operations. Implementations may support additional features (GPU acceleration,
    quantization, etc.) but must satisfy these core methods.

    Performance expectations:
    - `add()`: O(n) where n is the number of vectors; may require training first
    - `search()`: O(k * log(n)) for approximate indexes; O(n) for exact search
    - Indexes are typically trained before adding vectors (not shown in protocol)

    Concurrency notes:
    - Index operations are not thread-safe; serialize access from multiple threads
    - GPU indexes may block the calling thread during kernel execution
    - For async code, run index operations in thread pools to avoid blocking

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Examples
    --------
    >>> import numpy as np
    >>> from kgfoundry.search_api.types import FaissIndexProtocol, VectorArray
    >>> class SimpleIndex:
    ...     def add(self, vectors: VectorArray) -> None:
    ...         self._vectors = vectors
    ...
    ...     def search(
    ...         self, vectors: VectorArray, k: int
    ...     ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    ...         # Simple inner product search
    ...         scores = vectors @ self._vectors.T
    ...         indices = np.argsort(-scores, axis=1)[:, :k]
    ...         distances = np.take_along_axis(scores, indices, axis=1)
    ...         return distances.astype(np.float32), indices.astype(np.int64)
    >>> idx: FaissIndexProtocol = SimpleIndex()

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def add(self, vectors: VectorArray) -> None:
        """Add vectors to the index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Array of shape (n_vectors, dimension) with float32 dtype.
            Vectors should be normalized to unit length for inner-product search.

        Raises
        ------
        RuntimeError
            If index has not been trained (for trainable indexes).
        ValueError
            If vector dimensions do not match index configuration.
        """
        ...

    def search(self, vectors: VectorArray, k: int) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search for nearest neighbors.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Query vectors of shape (n_queries, dimension) with float32 dtype.
            Must be normalized to unit length for inner-product search.
        k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        tuple[tuple[int, ...] | np.float32, tuple[int, ...] | np.int64]
            Tuple of (distances, indices) where:
            - distances: shape (n_queries, k) with similarity scores (higher is better for IP)
            - indices: shape (n_queries, k) with vector indices in the index

        Notes
        -----
        - For inner-product metrics, higher scores indicate better matches
        - Invalid indices (no match) are typically represented as -1
        - Search performance depends on index type (exact vs approximate)
        """
        ...

    def train(self, vectors: VectorArray) -> None:
        """Train the index with sample vectors (optional method).

        <!-- auto:docstring-builder v1 -->

        Some FAISS indexes (e.g., quantized indexes) require training before
        adding vectors. This method is optional - not all indexes support it.

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Training vectors of shape (n_train, dimension) with float32 dtype.

        Notes
        -----
        - Flat indexes (exact search) do not require training
        - Quantized indexes (IVF, PQ) require training before add()
        - Calling train() on a non-trainable index is a no-op
        """
        ...

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:
        """Add vectors with explicit IDs (optional method).

        <!-- auto:docstring-builder v1 -->

        Some FAISS indexes support adding vectors with explicit IDs rather than
        sequential indices. This method is optional - not all indexes support it.

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Vectors to add, shape (n_vectors, dimension) with float32 dtype.
        ids : tuple[int, ...] | np.int64
            Explicit IDs for each vector, shape (n_vectors,) with int64 dtype.

        Notes
        -----
        - IndexIDMap2 wrapper enables add_with_ids for any base index
        - If not supported, use add() which assigns sequential IDs
        """
        ...


class FaissModuleProtocol(Protocol):
    """Protocol for FAISS module surface used by adapters.

    <!-- auto:docstring-builder v1 -->

    This protocol describes the subset of FAISS API used by kgfoundry
    adapters. It enables type checking and allows fallback implementations
    (e.g., `_SimpleFaissModule`) to satisfy the protocol.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Examples
    --------
    >>> from kgfoundry.search_api.types import FaissModuleProtocol, FaissIndexProtocol
    >>> class MockFaissModule:
    ...     METRIC_INNER_PRODUCT = 1
    ...
    ...     def IndexFlatIP(self, dimension: int) -> FaissIndexProtocol: ...
    ...     def write_index(self, index: FaissIndexProtocol, path: str) -> None: ...
    ...     def read_index(self, path: str) -> FaissIndexProtocol: ...
    ...     def normalize_L2(self, vectors: VectorArray) -> None: ...
    >>> module: FaissModuleProtocol = MockFaissModule()

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    METRIC_INNER_PRODUCT: int
    """Constant for inner-product metric (used with index_factory)."""

    METRIC_L2: int
    """Constant for L2 distance metric (used with index_factory)."""

    def IndexFlatIP(self, dimension: int) -> FaissIndexProtocol:  # noqa: N802
        """Create a flat inner-product index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        dimension : int
            Vector dimension (must match training/query vectors).

        Returns
        -------
        FaissIndexProtocol
            A flat (exact) index supporting inner-product search.

        Notes
        -----
        - Flat indexes provide exact search but are slower for large datasets
        - Suitable for small-to-medium corpora (< 1M vectors)
        - No training required before adding vectors
        """
        ...

    def index_factory(self, dimension: int, factory_string: str, metric: int) -> FaissIndexProtocol:
        """Create an index from a factory string.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        dimension : int
            Vector dimension.
        factory_string : str
            Factory description (e.g., "IVF8192,PQ64" for quantized index).
        metric : int
            Metric type (METRIC_INNER_PRODUCT or METRIC_L2).

        Returns
        -------
        FaissIndexProtocol
            Configured index instance.

        Examples
        --------
        >>> # Create quantized index for large-scale search
        >>> index = faiss.index_factory(2560, "OPQ64,IVF8192,PQ64", faiss.METRIC_INNER_PRODUCT)
        """
        ...

    def IndexIDMap2(self, index: FaissIndexProtocol) -> FaissIndexProtocol:  # noqa: N802
        """Wrap an index with 64-bit ID mapping.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        index : FaissIndexProtocol
            Base index to wrap.

        Returns
        -------
        FaissIndexProtocol
            Index with ID mapping support (supports add_with_ids).

        Notes
        -----
        - Use IndexIDMap2 for large corpora requiring 64-bit IDs
        - Wrapped index supports add_with_ids(vectors, ids) method
        """
        ...

    def write_index(self, index: FaissIndexProtocol, path: str) -> None:
        """Persist an index to disk.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        index : FaissIndexProtocol
            Index instance to save.
        path : str
            File path for the persisted index.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        ...

    def read_index(self, path: str) -> FaissIndexProtocol:
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
        FileNotFoundError
            If the index file does not exist.
        OSError
            If the file cannot be read or is corrupted.
        """
        ...

    def normalize_L2(self, vectors: VectorArray) -> None:  # noqa: N802
        """Normalize vectors to unit length in-place.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        vectors : tuple[int, ...] | np.float32
            Array to normalize (modified in-place).

        Notes
        -----
        - Normalization is required for inner-product search
        - Operation is destructive (modifies input array)
        - Zero vectors are handled gracefully (no division by zero)
        """
        ...


class GpuResourcesProtocol(Protocol):
    """Protocol for FAISS GPU resources.

    <!-- auto:docstring-builder v1 -->

    This protocol describes the StandardGpuResources interface used for GPU
    index operations. Implementations manage GPU memory and resources.

    Examples
    --------
    >>> from kgfoundry.search_api.types import GpuResourcesProtocol
    >>> class MockGpuResources:
    ...     def __init__(self) -> None:
    ...         pass
    >>> resources: GpuResourcesProtocol = MockGpuResources()
    """

    def __init__(self) -> None:
        """Initialize GPU resources.

        <!-- auto:docstring-builder v1 -->
        """
        ...


class GpuClonerOptionsProtocol(Protocol):
    """Protocol for FAISS GPU cloner options.

    <!-- auto:docstring-builder v1 -->

    This protocol describes the GpuClonerOptions interface used when cloning
    CPU indexes to GPU. The `use_cuvs` attribute controls cuVS acceleration.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Examples
    --------
    >>> from kgfoundry.search_api.types import GpuClonerOptionsProtocol
    >>> class MockGpuClonerOptions:
    ...     use_cuvs: bool = False
    >>> options: GpuClonerOptionsProtocol = MockGpuClonerOptions()
    >>> options.use_cuvs = True

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    use_cuvs: bool
    """Enable cuVS acceleration for GPU operations (default: False)."""


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Typed result from vector search operations.

    <!-- auto:docstring-builder v1 -->

    This dataclass represents a single search result with typed fields
    and performance metadata. Results are immutable to prevent accidental
    modification.

    Parameters
    ----------
    doc_id : str
        Describe ``doc_id``.
    chunk_id : str
        Describe ``chunk_id``.
    score : float
        Describe ``score``.
    vector_score : float
        Describe ``vector_score``.

    Examples
    --------
    >>> from kgfoundry.search_api.types import VectorSearchResult
    >>> result = VectorSearchResult(
    ...     doc_id="urn:doc:abc123",
    ...     chunk_id="urn:chunk:abc123:0-500",
    ...     score=0.95,
    ...     vector_score=0.95,
    ... )
    >>> assert result.score == 0.95
    """

    doc_id: str
    """Document identifier (URN format)."""

    chunk_id: str
    """Chunk identifier within the document."""

    score: float
    """Final relevance score (may combine multiple signals)."""

    vector_score: float
    """Raw vector similarity score (inner product or L2 distance)."""

    def __post_init__(self) -> None:
        """Validate result fields.

        <!-- auto:docstring-builder v1 -->

        This method is called automatically by dataclass after initialization.
        In frozen dataclasses, we can't modify fields, so validation is limited.
        """
        # Dataclass types are enforced at construction time, so runtime validation
        # is primarily for documentation. The type checker ensures correct usage.


class SpladeEncoderProtocol(Protocol):
    """Protocol for SPLADE encoder implementations.

    <!-- auto:docstring-builder v1 -->

    SPLADE (Sparse Lexical and Expansion) encoders transform text into
    sparse vector representations suitable for semantic search. This protocol
    defines the interface required for SPLADE-based search operations.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Examples
    --------
    >>> from kgfoundry.search_api.types import SpladeEncoderProtocol, VectorArray
    >>> import numpy as np
    >>> class MySpladeEncoder:
    ...     def encode(self, texts: Sequence[str]) -> VectorArray:
    ...         # Return sparse vectors for input texts
    ...         return np.array([[0.1, 0.0, 0.5]], dtype=np.float32)
    >>> encoder: SpladeEncoderProtocol = MySpladeEncoder()
    >>> vecs = encoder.encode(["query text"])
    >>> assert vecs.shape[0] == 1

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def encode(self, texts: Sequence[str]) -> VectorArray:
        """Encode text sequences into sparse vector representations.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        texts : str
            Input text sequences to encode.

        Returns
        -------
        tuple[int, ...] | np.float32
            Sparse vector array of shape (len(texts), vocab_size) with float32 dtype.
            Vectors are typically sparse (many zeros) and represent term importance.

        Raises
        ------
        RuntimeError
            If the encoder model fails to load or encode.
        ValueError
            If input texts are invalid or exceed maximum sequence length.
        """
        ...


class BM25IndexProtocol(Protocol):
    """Protocol for BM25 lexical search index implementations.

    <!-- auto:docstring-builder v1 -->

    BM25 indexes provide term-frequency based lexical search over document
    collections. This protocol defines the interface required for BM25-based
    search operations.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Examples
    --------
    >>> from kgfoundry.search_api.types import BM25IndexProtocol
    >>> class MyBM25Index:
    ...     def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
    ...         # Return (doc_id, score) tuples
    ...         return [("doc1", 0.95), ("doc2", 0.87)]
    >>> index: BM25IndexProtocol = MyBM25Index()
    >>> results = index.search("search query", k=5)
    >>> assert len(results) <= 5

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search the index for documents matching the query.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        query : str
            Search query string (will be tokenized internally).
        k : int, optional
            Maximum number of results to return.
            Defaults to ``10``.

        Returns
        -------
        list[tuple[str, float]]
            List of (document_id, score) tuples, sorted by score descending.
            Scores are BM25 relevance scores (higher is better).

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        ValueError
            If the query is empty or invalid.
        """
        ...


@dataclass(frozen=True, slots=True)
class AgentSearchQuery:
    """Query parameters for agent catalog search operations.

    <!-- auto:docstring-builder v1 -->

    This dataclass represents a search query with typed fields for query text,
    result count, filtering facets, and explanation flags. All fields are
    validated at construction time.

    Parameters
    ----------
    query : str
        Describe ``query``.
    k : int, optional
        Describe ``k``.
        Defaults to ``10``.
    facets : str | str | NoneType, optional
        Describe ``facets``.
        Defaults to ``None``.
    explain : bool, optional
        Describe ``explain``.
        Defaults to ``False``.

    Examples
    --------
    >>> from kgfoundry.search_api.types import AgentSearchQuery
    >>> query = AgentSearchQuery(
    ...     query="vector store",
    ...     k=10,
    ...     facets={"package": "search_api"},
    ...     explain=True,
    ... )
    >>> assert query.k == 10
    >>> assert query.facets["package"] == "search_api"
    """

    query: str
    """Search query text (tokenized and normalized internally)."""

    k: int = 10
    """Maximum number of results to return.

    Must be positive and typically bounded (e.g., 1 <= k <= 100).
    """

    facets: Mapping[str, str] | None = None
    """Optional facet filters for narrowing search results.

    Common facets include: package, module, kind, stability, deprecated.
    Values are matched exactly (case-sensitive).
    """

    explain: bool = False
    """Whether to include explanation metadata in results.

    When True, results include detailed scoring breakdowns and match
    highlights for debugging and transparency.
    """

    def __post_init__(self) -> None:
        """Validate query parameters.

        <!-- auto:docstring-builder v1 -->

        Raises
        ------
        ValueError
            If query is empty or k is not positive.
        """
        if not self.query.strip():
            msg = "Query text cannot be empty"
            raise ValueError(msg)
        if self.k <= 0:
            msg = f"k must be positive, got {self.k}"
            raise ValueError(msg)


class VectorSearchResultTypedDict(TypedDict, total=True):
    """Describe VectorSearchResultTypedDict.

    &lt;!-- auto:docstring-builder v1 --&gt;

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
    """

    symbol_id: str
    """Fully qualified symbol identifier (e.g., 'py:module.Class.method')."""

    score: float
    """Final relevance score combining lexical and vector signals."""

    lexical_score: float
    """BM25 lexical search score (0.0 to 1.0)."""

    vector_score: float
    """Vector similarity score from dense/sparse search (0.0 to 1.0)."""

    package: str
    """Package name containing the symbol."""

    module: str
    """Module name containing the symbol."""

    qname: str
    """Qualified name of the symbol within its module."""

    kind: str
    """Symbol kind (e.g., 'class', 'function', 'module')."""

    anchor: Mapping[str, int | None]
    """Source anchor metadata (start_line, end_line, etc.)."""

    metadata: Mapping[str, JsonValue]
    """Additional metadata (stability, deprecated, summary, etc.)."""


class AgentSearchResponse(TypedDict, total=True):
    """Describe AgentSearchResponse.

    &lt;!-- auto:docstring-builder v1 --&gt;

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
    """

    results: list[VectorSearchResultTypedDict]
    """List of search results, sorted by score descending."""

    total: int
    """Total number of results (may exceed len(results) if truncated)."""

    took_ms: int
    """Query execution time in milliseconds."""

    metadata: Mapping[str, JsonValue]
    """Response metadata (alpha, backend, query_info, etc.)."""
