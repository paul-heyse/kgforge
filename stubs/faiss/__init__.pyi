from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = ["METRIC_INNER_PRODUCT", "GpuClonerOptions", "StandardGpuResources"]

type VectorArray = npt.NDArray[np.float32]
type SearchResult = tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]

class VectorIndex(Protocol):
    """Protocol capturing the FAISS vector index behaviour used in kgfoundry."""

    dimension: int

    def add(self, vectors: VectorArray) -> None: ...
    def search(self, vectors: VectorArray, count: int) -> SearchResult: ...

class IndexIDMap2(VectorIndex):
    dimension: int

    def __init__(self, index: VectorIndex) -> None: ...
    def add_with_ids(self, vectors: VectorArray, ids: npt.NDArray[np.int64]) -> None: ...

class StandardGpuResources:
    """Stub FAISS GPU resource."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

class GpuClonerOptions:
    """Stub FAISS GPU cloner options."""

    use_cuvs: bool

    def __init__(self, *args: object, **kwargs: object) -> None: ...

METRIC_INNER_PRODUCT: int

def index_factory(dimension: int, description: str, metric: int) -> VectorIndex: ...
def index_cpu_to_gpu(
    resources: StandardGpuResources, device: int, index: VectorIndex
) -> VectorIndex: ...
def normalize_L2(vectors: VectorArray) -> None: ...  # noqa: N802
def __getattr__(name: str) -> object: ...
