"""Type stubs for FAISS library.

This module provides type hints for FAISS operations used in kgfoundry.
The stubs are minimal but sufficient for type checking vector search
operations.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "METRIC_INNER_PRODUCT",
    "METRIC_L2",
    "GpuClonerOptions",
    "GpuParameterSpace",
    "IndexIDMap2",
    "ParameterSpace",
    "StandardGpuResources",
    "VectorIndex",
    "index_cpu_to_gpu",
    "index_factory",
    "index_gpu_to_cpu",
    "normalize_L2",
    "read_index",
    "write_index",
]

type VectorArray = npt.NDArray[np.float32]
type SearchResult = tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]

class VectorIndex(Protocol):
    """Protocol capturing the FAISS vector index behaviour used in kgfoundry."""

    dimension: int

    def add(self, vectors: VectorArray) -> None:
        """Add vectors to the index."""
        ...

    def search(self, vectors: VectorArray, count: int) -> SearchResult:
        """Search for nearest neighbors."""
        ...

    def train(self, vectors: VectorArray) -> None:
        """Train the index (for trainable indexes)."""
        ...

class IndexIDMap2(VectorIndex):
    """Index wrapper with 64-bit ID mapping."""

    dimension: int

    def __init__(self, index: VectorIndex) -> None:
        """Wrap an index with ID mapping."""

    def add_with_ids(self, vectors: VectorArray, ids: npt.NDArray[np.int64]) -> None:
        """Add vectors with custom IDs."""

class StandardGpuResources:
    """Stub FAISS GPU resource manager."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize GPU resources."""

class GpuClonerOptions:
    """Stub FAISS GPU cloner options."""

    use_cuvs: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize GPU cloner options."""

class ParameterSpace:
    """Stub for CPU parameter space."""

    def set_index_parameter(self, index: VectorIndex, name: str, value: object) -> None:
        """Set an index parameter."""

class GpuParameterSpace:
    """Stub for GPU parameter space."""

    def set_index_parameter(self, index: VectorIndex, name: str, value: object) -> None:
        """Set an index parameter."""

METRIC_INNER_PRODUCT: int
"""Constant for inner-product metric."""

METRIC_L2: int
"""Constant for L2 distance metric."""

def index_factory(dimension: int, description: str, metric: int) -> VectorIndex:
    """Create an index from a factory string."""

def index_cpu_to_gpu(
    resources: StandardGpuResources,
    device: int,
    index: VectorIndex,
    options: GpuClonerOptions | None = None,
) -> VectorIndex:
    """Clone a CPU index to GPU."""

def index_gpu_to_cpu(index: VectorIndex) -> VectorIndex:
    """Convert a GPU index back to CPU."""

def normalize_L2(vectors: VectorArray) -> None:  # noqa: N802
    """Normalize vectors to unit length in-place."""

def write_index(index: VectorIndex, path: str) -> None:
    """Write an index to disk."""

def read_index(path: str) -> VectorIndex:
    """Read an index from disk."""

def __getattr__(name: str) -> object:
    """Fallback for dynamic attributes."""
