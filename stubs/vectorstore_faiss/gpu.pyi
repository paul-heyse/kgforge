"""Stub for vectorstore_faiss.gpu with typed array aliases and GPU-aware facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from search_api.types import FaissIndexProtocol, FaissModuleProtocol

type FloatArray = npt.NDArray[np.float32]
"""Type alias for float32 arrays used in FAISS operations."""

type IntArray = npt.NDArray[np.int64]
"""Type alias for int64 arrays used in FAISS operations."""

type StrArray = npt.NDArray[np.str_]
"""Type alias for string arrays used in ID mappings."""

type VecArray = npt.NDArray[np.float32]
"""Type alias for vector arrays compatible with FAISS indexes."""

__all__: Final[list[str]]

class FaissGpuIndex:
    """GPU-aware FAISS index facade with fallback to CPU.

    Parameters
    ----------
    faiss_module : FaissModuleProtocol
        FAISS module providing factory and search operations.
    factory : str
        FAISS factory string (e.g., "OPQ64,IVF8192,PQ64").
    metric : str
        Metric type: "ip" (inner product) or "l2" (L2 distance).
    nprobe : int, optional
        Search probe parameter for IVF indexes. Defaults to 64.
    use_gpu : bool, optional
        Enable GPU acceleration when available. Defaults to True.
    use_cuvs : bool, optional
        Enable cuVS acceleration. Defaults to True.
    devices : Sequence[int], optional
        GPU device IDs. Defaults to (0,).
    """

    faiss_module: FaissModuleProtocol
    factory: str
    metric: str
    nprobe: int
    use_gpu: bool
    use_cuvs: bool
    devices: Sequence[int]

    def __init__(
        self,
        faiss_module: FaissModuleProtocol,
        factory: str,
        metric: str,
        *,
        nprobe: int = 64,
        use_gpu: bool = True,
        use_cuvs: bool = True,
        devices: Sequence[int] = (0,),
    ) -> None: ...
    def prepare(self, trained_index: FaissIndexProtocol) -> FaissIndexProtocol:
        """Clone index to GPU and configure search parameters.

        Parameters
        ----------
        trained_index : FaissIndexProtocol
            CPU-based trained FAISS index.

        Returns
        -------
        FaissIndexProtocol
            Index optimized for GPU or CPU (fallback).
        """
        ...

    def search(self, query: FloatArray, k: int) -> list[tuple[int, float]]:
        """Search for top-k nearest neighbors.

        Parameters
        ----------
        query : FloatArray
            Query vector of shape (dimension,).
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[int, float]]
            Sorted list of (doc_id, score) pairs.
        """
        ...
