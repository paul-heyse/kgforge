"""Stub for search_api.faiss_adapter with typed array aliases and FAISS adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

type FloatArray = npt.NDArray[np.float32]
"""Type alias for float32 arrays used in FAISS operations."""

type IntArray = npt.NDArray[np.int64]
"""Type alias for int64 arrays used in FAISS operations."""

type StrArray = npt.NDArray[np.str_]
"""Type alias for string arrays used in ID mappings."""

type VecArray = npt.NDArray[np.float32]
"""Type alias for vector arrays compatible with FAISS indexes."""

__all__: Final[list[str]]

HAVE_FAISS: bool
"""Whether FAISS library is available."""

MIN_FACTORY_DIMENSION: Final[int]

class DenseVecs:
    """Dense vector matrix and ID mapping for FAISS indexes.

    Parameters
    ----------
    ids : list[str]
        Document/chunk identifiers.
    matrix : FloatArray
        Matrix of shape (n_vectors, embedding_dim).
    """

    ids: list[str]
    matrix: FloatArray

class FaissAdapter:
    """Build and search FAISS indexes with optional GPU acceleration.

    Parameters
    ----------
    db_path : str
        Path to DuckDB registry or Parquet vectors.
    factory : str, optional
        FAISS factory string. Defaults to "OPQ64,IVF8192,PQ64".
    metric : str, optional
        Metric type: "ip" or "l2". Defaults to "ip".
    nprobe : int, optional
        IVF search parameter. Defaults to 64.
    use_gpu : bool, optional
        Enable GPU acceleration. Defaults to True.
    use_cuvs : bool, optional
        Enable cuVS acceleration. Defaults to True.
    gpu_devices : Sequence[int] | None, optional
        GPU device IDs. Defaults to (0,).
    """

    db_path: str
    factory: str
    metric: str
    nprobe: int
    use_gpu: bool
    use_cuvs: bool
    index: object  # FaissIndexProtocol | None
    idmap: list[str] | None
    vecs: DenseVecs | None

    def __init__(
        self,
        db_path: str,
        *,
        factory: str = "OPQ64,IVF8192,PQ64",
        metric: str = "ip",
        nprobe: int = 64,
        use_gpu: bool = True,
        use_cuvs: bool = True,
        gpu_devices: Sequence[int] | None = None,
    ) -> None: ...
    def build(self) -> None:
        """Build FAISS index from persisted vectors."""
        ...

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Load existing index or rebuild from vectors."""
        ...

    def search(
        self, query: Sequence[float] | NDArray[np.float32], k: int
    ) -> list[tuple[str, float]]:
        """Search for top-k nearest neighbors.

        Parameters
        ----------
        query : Sequence[float] | NDArray[np.float32]
            Query vector.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[str, float]]
            List of (doc_id, score) pairs.
        """
        ...

    def save(self, index_uri: str, idmap_uri: str | None = None) -> None:
        """Persist index and ID mapping to disk."""
        ...
