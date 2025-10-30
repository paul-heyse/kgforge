"""Overview of faiss adapter.

This module bundles faiss adapter logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

import duckdb
import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap

__all__ = ["DenseVecs", "FaissAdapter", "VecArray"]

__navmap__: Final[NavMap] = {
    "title": "search_api.faiss_adapter",
    "synopsis": "Dense retrieval utilities that wrap FAISS with DuckDB persistence.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["DenseVecs", "FaissAdapter", "VecArray"],
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        "DenseVecs": {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        },
        "FaissAdapter": {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        },
        "VecArray": {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        },
    },
}

MIN_FACTORY_DIMENSION: Final[int] = 64

faiss: ModuleType | None = None

try:
    try:
        from libcuvs import load_library as _load_cuvs

        _load_cuvs()
    except Exception:
        pass
    import faiss

    HAVE_FAISS = True
except Exception:  # pragma: no cover - exercised when FAISS is unavailable
    faiss = None
    HAVE_FAISS = False


# [nav:anchor VecArray]
type VecArray = NDArray[np.float32]
type IndexArray = NDArray[np.int64]
type FloatArrayLike = NDArray[np.floating[Any]]


# [nav:anchor DenseVecs]
@dataclass
class DenseVecs:
    """Store dense vectors alongside their document identifiers.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    ids : list[str]
        Describe ``ids``.
    mat : VecArray
        Describe ``mat``.
"""

    ids: list[str]
    mat: VecArray


# [nav:anchor FaissAdapter]
class FaissAdapter:
    """Manage FAISS vector indexes with optional GPU acceleration.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    db_path : str
        Describe ``db_path``.
    factory : str, optional
        Describe ``factory``.
        Defaults to ``'OPQ64,IVF8192,PQ64'``.
    metric : str, optional
        Describe ``metric``.
        Defaults to ``'ip'``.
"""

    def __init__(
        self,
        db_path: str,
        factory: str = "OPQ64,IVF8192,PQ64",
        metric: str = "ip",
    ) -> None:
        """Initialise the adapter with storage paths and FAISS configuration.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        db_path : str
            Describe ``db_path``.
        factory : str, optional
            Describe ``factory``.
            Defaults to ``'OPQ64,IVF8192,PQ64'``.
        metric : str, optional
            Describe ``metric``.
            Defaults to ``'ip'``.
"""
        self.db_path = db_path
        self.factory = factory
        self.metric = metric
        self.index: Any | None = None
        self.idmap: list[str] | None = None
        self.vecs: DenseVecs | None = None

    def _load_dense_from_parquet(self, source: Path) -> DenseVecs:
        """Load and normalise dense vectors stored in a Parquet file.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        source : Path
            Describe ``source``.
            

        Returns
        -------
        DenseVecs
            Normalised vectors paired with their document identifiers.
            
            
            

        Raises
        ------
        RuntimeError
        Raised when the file does not contain any vector records.
"""
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"""
                SELECT chunk_id, vector
                FROM read_parquet('{source}', union_by_name=true)
            """
            ).fetchall()
        finally:
            con.close()
        if not rows:
            message = f"No dense vectors discovered in {source}"
            raise RuntimeError(message)
        ids = [row[0] for row in rows]
        mat = np.stack([np.asarray(row[1], dtype=np.float32) for row in rows]).astype(
            np.float32,
            copy=False,
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        normalized = cast(VecArray, mat / norms)
        return DenseVecs(ids=ids, mat=normalized)

    def _load_dense_parquet(self) -> DenseVecs:
        """Load dense vectors from the adapter's managed Parquet dataset.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        DenseVecs
            Normalised vectors available to the adapter.
            
            
            

        Raises
        ------
        RuntimeError
        Raised when no vectors are available on disk.
"""
        candidate = Path(self.db_path)
        if candidate.is_dir() or candidate.suffix == ".parquet":
            return self._load_dense_from_parquet(candidate)
        if not candidate.exists():
            message = "DuckDB registry not found"
            raise RuntimeError(message)
        try:
            con = duckdb.connect(self.db_path)
        except duckdb.Error:
            return self._load_dense_from_parquet(candidate)
        try:
            dense_run = con.execute(
                "SELECT parquet_root, dim FROM dense_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not dense_run:
                message = "No dense_runs found"
                raise RuntimeError(message)
            root = dense_run[0]
            rows = con.execute(
                f"""
                SELECT chunk_id, vector
                FROM read_parquet('{root}/*/*.parquet', union_by_name=true)
            """
            ).fetchall()
        finally:
            con.close()
        ids = [row[0] for row in rows]
        mat = np.stack([np.asarray(row[1], dtype=np.float32) for row in rows]).astype(
            np.float32,
            copy=False,
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        normalized = cast(VecArray, mat / norms)
        return DenseVecs(ids=ids, mat=normalized)

    def build(self) -> None:
        """Build or refresh the FAISS index from the persisted vector store.

        <!-- auto:docstring-builder v1 -->

        Raises
        ------
        RuntimeError
        Raised when dense vectors cannot be loaded from disk.
"""
        vectors = self._load_dense_parquet()
        self.vecs = vectors
        self.idmap = vectors.ids

        faiss_module = faiss
        if not HAVE_FAISS or faiss_module is None:
            return
        dimension = vectors.mat.shape[1]
        metric_type = (
            faiss_module.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss_module.METRIC_L2
        )
        factory = self.factory if dimension >= MIN_FACTORY_DIMENSION else "Flat"
        cpu = faiss_module.index_factory(dimension, factory, metric_type)
        cpu = faiss_module.IndexIDMap2(cpu)
        train = vectors.mat[: min(100000, vectors.mat.shape[0])].copy()
        faiss_module.normalize_L2(train)
        index_map = cast(Any, cpu)
        index_map.train(train)
        ids64 = cast(IndexArray, np.arange(vectors.mat.shape[0], dtype=np.int64))
        index_map.add_with_ids(vectors.mat, ids64)
        self.index = self._clone_to_gpu(cpu)

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Load an existing index or build one if no cached artefact is present.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        cpu_index_path : str | None, optional
            Describe ``cpu_index_path``.
            Defaults to ``None``.
"""
        faiss_module = faiss
        if faiss_module is None:
            self.build()
            return
        try:
            if HAVE_FAISS and cpu_index_path and Path(cpu_index_path).exists():
                cpu = faiss_module.read_index(cpu_index_path)
                self.index = self._clone_to_gpu(cpu)
                dense = self._load_dense_parquet()
                self.vecs = dense
                self.idmap = dense.ids
                return
        except Exception:
            pass
        self.build()

    def _clone_to_gpu(self, cpu_index: object) -> object:
        """Clone a CPU index onto a GPU when GPU support is available.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        cpu_index : object
            Describe ``cpu_index``.
            

        Returns
        -------
        object
            GPU-backed FAISS index when cloning succeeds, otherwise the original CPU index.
"""
        faiss_module = faiss
        if faiss_module is None:
            return cpu_index
        standard_resources = getattr(faiss_module, "StandardGpuResources", None)
        gpu_cloner_options = getattr(faiss_module, "GpuClonerOptions", None)
        index_cpu_to_gpu = getattr(faiss_module, "index_cpu_to_gpu", None)
        if standard_resources is None or gpu_cloner_options is None or index_cpu_to_gpu is None:
            return cpu_index
        resources = standard_resources()
        options = gpu_cloner_options()
        options.use_cuvs = True
        try:
            return index_cpu_to_gpu(resources, 0, cpu_index, options)
        except Exception:  # pragma: no cover - fallback path without cuVS
            options.use_cuvs = False
            return index_cpu_to_gpu(resources, 0, cpu_index, options)

    def search(self, qvec: FloatArrayLike, k: int = 10) -> list[list[tuple[str, float]]]:
        """Search the index for nearest neighbours.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        qvec : FloatArrayLike
            Describe ``qvec``.
        k : int, optional
            Describe ``k``.
            Defaults to ``10``.
            

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked matches for each query with document identifiers and similarity scores.
"""
        if self.vecs is None and self.index is None:
            return []
        queries = self._prepare_queries(qvec)
        if HAVE_FAISS and self.index is not None:
            return self._search_with_faiss(queries, k)
        return self._search_with_cpu(queries, k)

    def _prepare_queries(self, qvec: VecArray) -> VecArray:
        """Normalise query vectors to match FAISS search expectations.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        qvec : VecArray
            Describe ``qvec``.
            

        Returns
        -------
        VecArray
            Two-dimensional, normalised query matrix.
"""
        query_arr: VecArray = np.asarray(qvec, dtype=np.float32, order="C")
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
        return query_arr

    def _search_with_faiss(self, queries: VecArray, k: int) -> list[list[tuple[str, float]]]:
        """Perform a batched search using the GPU-backed FAISS index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        queries : VecArray
            Describe ``queries``.
        k : int
            Describe ``k``.
            

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked matches for each query.
            
            
            

        Raises
        ------
        RuntimeError
        Raised when the FAISS index or identifier mapping has not been loaded.
"""
        if self.index is None or self.idmap is None:
            message = "FAISS index or ID mapping not loaded"
            raise RuntimeError(message)
        distances, indices = self.index.search(queries, k)
        batches: list[list[tuple[str, float]]] = []
        for idx_row, dist_row in zip(indices, distances, strict=False):
            results: list[tuple[str, float]] = []
            for idx, score in zip(idx_row, dist_row, strict=False):
                if idx < 0:
                    continue
                results.append((self.idmap[int(idx)], float(score)))
            batches.append(results)
        return batches

    def _search_with_cpu(self, queries: VecArray, k: int) -> list[list[tuple[str, float]]]:
        """Perform a batched search using the CPU fallback implementation.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        queries : VecArray
            Describe ``queries``.
        k : int
            Describe ``k``.
            

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked matches for each query.
            
            
            

        Raises
        ------
        RuntimeError
        Raised when dense vectors have not been loaded into memory.
"""
        if self.vecs is None:
            message = "Dense vectors not loaded"
            raise RuntimeError(message)
        matrix = self.vecs.mat
        batches: list[list[tuple[str, float]]] = []
        for row in queries:
            normalised = row.copy()
            normalised /= np.linalg.norm(normalised) + 1e-9
            sims = matrix @ normalised
            topk = np.argsort(-sims)[:k]
            batches.append([(self.vecs.ids[i], float(sims[i])) for i in topk])
        return batches
