"""Type stubs for FAISS library.

This module provides type hints for FAISS operations used in kgfoundry.
The stubs expose typed APIs aligned with search_api.types protocols,
enabling full type checking for vector search operations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# Re-export Protocol types from search_api.types for convenience
from search_api.types import FaissIndexProtocol

__all__ = [
    "METRIC_INNER_PRODUCT",
    "METRIC_L2",
    "GpuClonerOptions",
    "GpuDistanceParams",
    "GpuIndexCagraConfig",
    "GpuIndexFlatConfig",
    "GpuIndexFlatIP",
    "GpuIndexIVFConfig",
    "GpuParameterSpace",
    "IndexFlatIP",
    "IndexIDMap2",
    "ParameterSpace",
    "StandardGpuResources",
    "VectorIndex",
    "index_cpu_to_gpu",
    "index_cpu_to_gpu_multiple",
    "index_factory",
    "index_gpu_to_cpu",
    "knn_gpu",
    "normalize_L2",
    "read_index",
    "read_index_binary",
    "serialize_index",
    "should_use_cuvs",
    "write_index",
    "write_index_binary",
]

type VectorArray = npt.NDArray[np.float32]
type IndexArray = npt.NDArray[np.int64]
type SearchResult = tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]

class VectorIndex(FaissIndexProtocol):
    """Protocol capturing the FAISS vector index behaviour used in kgfoundry.

    This protocol extends FaissIndexProtocol from search_api.types and
    provides backward compatibility. Prefer using FaissIndexProtocol directly.
    """

    dimension: int
    """Vector dimension for this index."""

    def add(self, vectors: VectorArray) -> None:
        """Add vectors to the index."""
        ...

    def search(self, vectors: VectorArray, k: int) -> SearchResult:
        """Search for nearest neighbors.

        Parameters
        ----------
        vectors : VectorArray
            Query vectors of shape (n_queries, dimension).
        k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        SearchResult
            Tuple of (distances, indices) arrays.
        """
        ...

    def train(self, vectors: VectorArray) -> None:
        """Train the index (for trainable indexes)."""
        ...

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:
        """Add vectors with custom IDs (optional method).

        Not all indexes support this method. Use IndexIDMap2 wrapper
        to enable add_with_ids for any base index.
        """
        ...

class IndexIDMap2(VectorIndex):
    """Index wrapper with 64-bit ID mapping."""

    dimension: int

    def __init__(self, index: VectorIndex) -> None:
        """Wrap an index with ID mapping."""
        ...

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:
        """Add vectors with custom IDs.

        Parameters
        ----------
        vectors : VectorArray
            Vectors to add.
        ids : IndexArray
            64-bit integer IDs corresponding to each vector.
        """
        ...

class IndexFlatIP(VectorIndex):
    """Flat inner-product index."""

    dimension: int

    def __init__(self, dimension: int) -> None:
        """Create a flat inner-product index.

        Parameters
        ----------
        dimension : int
            Vector dimension.
        """
        ...

    def add_with_ids(self, vectors: VectorArray, ids: IndexArray) -> None:
        """Flat indexes do not support add_with_ids directly.

        Use IndexIDMap2 wrapper to enable add_with_ids for flat indexes.
        This method is provided for Protocol compatibility but will raise
        NotImplementedError if called directly.
        """
        ...

class StandardGpuResources:
    """Stub FAISS GPU resource manager."""

    def __init__(self) -> None:
        """Initialize GPU resources."""
        ...

    def setTempMemory(self, value: int) -> None:  # noqa: N802
        """Configure temporary GPU memory budget (bytes)."""
        ...

    def setPinnedMemory(self, value: int) -> None:  # noqa: N802
        """Configure pinned host memory budget (bytes)."""
        ...

class GpuIndexFlatIP(VectorIndex):
    """GPU-backed flat inner-product index."""

    dimension: int

    def __init__(
        self,
        resources: StandardGpuResources,
        dimension: int,
        config: object | None = ...,
    ) -> None:
        """Create a GPU flat inner-product index."""
        ...

class GpuIndexFlatConfig:
    """Configuration surface for GPU flat indexes."""

    use_cuvs: bool
    device: int

    def __init__(self) -> None:
        """Initialise configuration with default settings."""
        ...

class GpuIndexIVFConfig:
    """Configuration surface for GPU IVF indexes."""

    use_cuvs: bool
    device: int

    def __init__(self) -> None:
        """Initialise configuration with default settings."""
        ...

class GpuIndexCagraConfig:
    """Configuration surface for GPU CAGRA indexes."""

    use_cuvs: bool
    device: int

    def __init__(self) -> None:
        """Initialise configuration with default settings."""
        ...

class GpuClonerOptions:
    """Stub FAISS GPU cloner options."""

    use_cuvs: bool
    """Whether to use cuVS (CUDA Vector Search) backend."""

    def __init__(self) -> None:
        """Initialize GPU cloner options."""
        ...

class ParameterSpace:
    """Stub for CPU parameter space."""

    def set_index_parameter(self, index: VectorIndex, name: str, value: object) -> None:
        """Set an index parameter.

        Parameters
        ----------
        index : VectorIndex
            Index to configure.
        name : str
            Parameter name.
        value : object
            Parameter value.
        """
        ...

class GpuParameterSpace:
    """Stub for GPU parameter space."""

    def set_index_parameter(self, index: VectorIndex, name: str, value: object) -> None:
        """Set an index parameter.

        Parameters
        ----------
        index : VectorIndex
            Index to configure.
        name : str
            Parameter name.
        value : object
            Parameter value.
        """
        ...

    def initialize(self, index: VectorIndex) -> None:
        """Bind the parameter space to ``index``."""
        ...

class GpuDistanceParams:
    """Parameters describing GPU brute-force distance searches."""

    metric: int
    k: int
    dims: int

    def __init__(self) -> None:
        """Initialise distance parameters with defaults."""
        ...

METRIC_INNER_PRODUCT: int
"""Constant for inner-product metric."""

METRIC_L2: int
"""Constant for L2 distance metric."""

def index_factory(dimension: int, description: str, metric: int) -> VectorIndex:
    """Create an index from a factory string.

    Parameters
    ----------
    dimension : int
        Vector dimension.
    description : str
        Factory description (e.g., "IVF8192,PQ64").
    metric : int
        Metric type (METRIC_INNER_PRODUCT or METRIC_L2).

    Returns
    -------
    VectorIndex
        Configured index instance.
    """
    ...

def index_cpu_to_gpu(
    resources: StandardGpuResources,
    device: int,
    index: VectorIndex,
    options: GpuClonerOptions | None = None,
) -> VectorIndex:
    """Clone a CPU index to GPU.

    Parameters
    ----------
    resources : StandardGpuResources
        GPU resource manager.
    device : int
        GPU device index (typically 0).
    index : VectorIndex
        CPU index to clone.
    options : GpuClonerOptions | None, optional
        Cloning options. Defaults to None.

    Returns
    -------
    VectorIndex
        GPU-backed index instance.
    """
    ...

def index_cpu_to_gpu_multiple(
    resources: StandardGpuResources,
    devices: Sequence[int],
    index: VectorIndex,
    options: GpuClonerOptions | None = None,
) -> list[VectorIndex]:
    """Clone a CPU index to multiple GPU devices."""
    ...

def index_gpu_to_cpu(index: VectorIndex) -> VectorIndex:
    """Convert a GPU index back to CPU.

    Parameters
    ----------
    index : VectorIndex
        GPU index to convert.

    Returns
    -------
    VectorIndex
        CPU index instance.
    """
    ...

def normalize_L2(vectors: VectorArray) -> None:  # noqa: N802
    """Normalize vectors to unit length in-place.

    Parameters
    ----------
    vectors : VectorArray
        Array to normalize (modified in-place).
    """
    ...

def write_index(index: VectorIndex, path: str) -> None:
    """Write an index to disk.

    Parameters
    ----------
    index : VectorIndex
        Index instance to save.
    path : str
        File path for the persisted index.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    ...

def read_index(path: str) -> VectorIndex:
    """Read an index from disk.

    Parameters
    ----------
    path : str
        File path to the persisted index.

    Returns
    -------
    VectorIndex
        Loaded index instance.

    Raises
    ------
    FileNotFoundError
        If the index file does not exist.
    OSError
        If the file cannot be read or is corrupted.
    """
    ...

def write_index_binary(index: VectorIndex, path: str) -> None:
    """Write an index to disk in binary format."""
    ...

def read_index_binary(path: str) -> VectorIndex:
    """Read an index persisted in binary format."""
    ...

def serialize_index(index: VectorIndex) -> bytes:
    """Serialize an index into a bytes payload."""
    ...

def should_use_cuvs(params: object) -> bool:
    """Return ``True`` when cuVS acceleration should be engaged for ``params``."""
    ...

def knn_gpu(
    resources: StandardGpuResources,
    queries: VectorArray,
    base: VectorArray,
    k: int,
    *,
    metric: int,
    use_cuvs: bool | None = None,
) -> SearchResult:
    """Run brute-force KNN on GPU, optionally enabling cuVS kernels."""
    ...

def __getattr__(name: str) -> object:
    """Fallback for dynamic attributes."""
    ...
