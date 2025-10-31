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

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "FaissIndexProtocol",
    "FaissModuleProtocol",
    "VectorArray",
    "VectorSearchResult",
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
VectorArray = NDArray[np.float32]  # type: ignore[misc]  # numpy dtype contains Any
"""Type alias for float32 vector arrays used in FAISS operations.

All vectors must be normalized to unit length for inner-product search.
Dimensions are typically 2560 for dense embeddings or configurable for
sparse representations.
"""


class FaissIndexProtocol(Protocol):
    """Protocol for FAISS-compatible vector indexes.

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
    """

    def add(self, vectors: VectorArray) -> None:
        """Add vectors to the index.

        Parameters
        ----------
        vectors : VectorArray
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

        Parameters
        ----------
        vectors : VectorArray
            Query vectors of shape (n_queries, dimension) with float32 dtype.
            Must be normalized to unit length for inner-product search.
        k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.int64]
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


class FaissModuleProtocol(Protocol):
    """Protocol for FAISS module surface used by adapters.

    This protocol describes the subset of FAISS API used by kgfoundry
    adapters. It enables type checking and allows fallback implementations
    (e.g., `_SimpleFaissModule`) to satisfy the protocol.

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
    """

    METRIC_INNER_PRODUCT: int
    """Constant for inner-product metric (used with index_factory)."""

    METRIC_L2: int
    """Constant for L2 distance metric (used with index_factory)."""

    def IndexFlatIP(self, dimension: int) -> FaissIndexProtocol:  # noqa: N802
        """Create a flat inner-product index.

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

        Parameters
        ----------
        vectors : VectorArray
            Array to normalize (modified in-place).

        Notes
        -----
        - Normalization is required for inner-product search
        - Operation is destructive (modifies input array)
        - Zero vectors are handled gracefully (no division by zero)
        """
        ...


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Typed result from vector search operations.

    This dataclass represents a single search result with typed fields
    and performance metadata. Results are immutable to prevent accidental
    modification.

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

        This method is called automatically by dataclass after initialization.
        In frozen dataclasses, we can't modify fields, so validation is limited.
        """
        # Dataclass types are enforced at construction time, so runtime validation
        # is primarily for documentation. The type checker ensures correct usage.
