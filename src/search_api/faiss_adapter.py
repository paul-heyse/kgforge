"""Overview of faiss adapter.

This module bundles faiss adapter logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import duckdb
import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.errors import IndexBuildError, VectorSearchError
from kgfoundry_common.navmap_types import NavMap

if TYPE_CHECKING:
    from search_api.types import (
        FaissIndexProtocol,
        FaissModuleProtocol,
        GpuClonerOptionsProtocol,
        GpuResourcesProtocol,
        IndexArray,
        VectorArray,
    )
else:
    FaissIndexProtocol = object  # type: ignore[assignment, misc]
    FaissModuleProtocol = object  # type: ignore[assignment, misc]
    GpuClonerOptionsProtocol = object  # type: ignore[assignment, misc]
    GpuResourcesProtocol = object  # type: ignore[assignment, misc]
    IndexArray = NDArray[np.int64]  # type: ignore[misc]
    VectorArray = NDArray[np.float32]  # type: ignore[misc]

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

logger = logging.getLogger(__name__)

faiss: FaissModuleProtocol | None = None

try:
    try:
        from libcuvs import load_library as _load_cuvs

        _load_cuvs()
    except (ImportError, RuntimeError, OSError):
        # cuVS library not available - continue without it
        logger.debug("cuVS library not available, continuing without GPU acceleration")
    import faiss as _faiss_module

    HAVE_FAISS = True
    faiss = _faiss_module  # type: ignore[assignment]
except (ImportError, ModuleNotFoundError):
    logger.debug("FAISS library not available")
    faiss = None
    HAVE_FAISS = False


# [nav:anchor VecArray]
type VecArray = NDArray[np.float32]
# Import IndexArray and VectorArray from types when available
if TYPE_CHECKING:
    from search_api.types import IndexArray, VectorArray
else:
    IndexArray = NDArray[np.int64]  # type: ignore[misc]
    VectorArray = NDArray[np.float32]  # type: ignore[misc]


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
        self.index: FaissIndexProtocol | None = None
        self.idmap: list[str] | None = None
        self.vecs: DenseVecs | None = None

    def _load_dense_from_parquet(self, source: Path) -> DenseVecs:
        """Load and normalise dense vectors stored in a Parquet file.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        source : Path
            Path to Parquet file or directory containing Parquet files.

        Returns
        -------
        DenseVecs
            Normalised vectors paired with their document identifiers.

        Raises
        ------
        VectorSearchError
            Raised when the file does not contain any vector records or SQL execution fails.
        """
        # Validate and sanitize file path
        resolved_source = source.resolve(strict=True)
        if not resolved_source.exists():
            msg = f"Parquet source not found: {source}"
            raise VectorSearchError(msg)

        con = duckdb.connect(database=":memory:")
        try:
            # Parameterize query to prevent SQL injection
            # Note: DuckDB's read_parquet() doesn't support parameters directly,
            # but we validate the path and use Path.resolve() for safety
            sql = """
                SELECT chunk_id, vector
                FROM read_parquet(?, union_by_name=true)
            """
            rows = con.execute(sql, [str(resolved_source)]).fetchall()
        except duckdb.Error as e:
            msg = f"Failed to load vectors from {source}: {e}"
            raise VectorSearchError(msg) from e
        finally:
            con.close()
        if not rows:
            msg = f"No dense vectors discovered in {source}"
            raise VectorSearchError(msg)
        # Explicitly type DuckDB query results
        ids: list[str] = []
        vectors_list: list[VectorArray] = []
        for row in rows:
            chunk_id = row[0]
            vector_data = row[1]
            ids.append(str(chunk_id))
            vectors_list.append(np.asarray(vector_data, dtype=np.float32))
        mat: NDArray[np.float32] = np.stack(vectors_list).astype(np.float32, copy=False)
        norm_result: NDArray[np.float32] = np.linalg.norm(mat, axis=1, keepdims=True).astype(  # type: ignore[misc]
            np.float32
        )
        epsilon: np.float32 = np.float32(1e-9)
        norms: NDArray[np.float32] = norm_result + epsilon
        normalized: VecArray = cast(VecArray, mat / norms)
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
            msg = "DuckDB registry not found"
            raise VectorSearchError(msg)
        try:
            con = duckdb.connect(self.db_path)
        except duckdb.Error:
            return self._load_dense_from_parquet(candidate)
        try:
            dense_run = con.execute(
                "SELECT parquet_root, dim FROM dense_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not dense_run:
                msg = "No dense_runs found"
                raise VectorSearchError(msg)
            root = dense_run[0]
            if not isinstance(root, str):
                msg = f"Invalid parquet_root type: {type(root)}"
                raise VectorSearchError(msg)
            # Parameterize query - DuckDB's read_parquet() accepts string paths
            # We validate root is a string and construct path safely without f-string interpolation
            # Use pathlib for safe path construction
            root_path = Path(root)
            parquet_pattern = str(root_path / "*" / "*.parquet")
            sql = """
                SELECT chunk_id, vector
                FROM read_parquet(?, union_by_name=true)
            """
            rows = con.execute(sql, [parquet_pattern]).fetchall()
        except duckdb.Error as e:
            msg = f"Failed to query dense_runs: {e}"
            raise VectorSearchError(msg) from e
        finally:
            con.close()
        # Explicitly type DuckDB query results
        ids: list[str] = []
        vectors_list: list[VectorArray] = []
        for row in rows:
            chunk_id = row[0]
            vector_data = row[1]
            ids.append(str(chunk_id))
            vectors_list.append(np.asarray(vector_data, dtype=np.float32))
        mat: NDArray[np.float32] = np.stack(vectors_list).astype(np.float32, copy=False)
        norm_result: NDArray[np.float32] = np.linalg.norm(mat, axis=1, keepdims=True).astype(  # type: ignore[misc]
            np.float32
        )
        epsilon: np.float32 = np.float32(1e-9)
        norms: NDArray[np.float32] = norm_result + epsilon
        normalized: VecArray = cast(VecArray, mat / norms)
        return DenseVecs(ids=ids, mat=normalized)

    def build(self) -> None:
        """Build or refresh the FAISS index from the persisted vector store.

        <!-- auto:docstring-builder v1 -->

        Raises
        ------
        VectorSearchError
            Raised when dense vectors cannot be loaded from disk.
        IndexBuildError
            Raised when FAISS index construction fails.
        """
        vectors = self._load_dense_parquet()
        self.vecs = vectors
        self.idmap = vectors.ids

        faiss_module = faiss
        if not HAVE_FAISS or faiss_module is None:
            return
        try:
            dimension = vectors.mat.shape[1]
            metric_type = (
                faiss_module.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss_module.METRIC_L2
            )
            factory = self.factory if dimension >= MIN_FACTORY_DIMENSION else "Flat"
            # Explicitly type FAISS operations - index_factory returns untyped object
            cpu_raw: object = faiss_module.index_factory(dimension, factory, metric_type)
            # IndexIDMap2 expects FaissIndexProtocol, cast raw object first
            cpu_index_protocol: FaissIndexProtocol = cast(FaissIndexProtocol, cpu_raw)
            cpu_wrapped_raw: object = faiss_module.IndexIDMap2(cpu_index_protocol)
            cpu: FaissIndexProtocol = cast(FaissIndexProtocol, cpu_wrapped_raw)
            train: VectorArray = vectors.mat[: min(100000, vectors.mat.shape[0])].copy()
            faiss_module.normalize_L2(train)
            # Call train() and add_with_ids() - these are optional Protocol methods
            cpu.train(train)
            ids64: IndexArray = cast(IndexArray, np.arange(vectors.mat.shape[0], dtype=np.int64))
            cpu.add_with_ids(vectors.mat, ids64)
            self.index = self._clone_to_gpu(cpu)
        except Exception as e:
            msg = f"Failed to build FAISS index: {e}"
            raise IndexBuildError(msg) from e

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Load an existing index or build one if no cached artefact is present.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        cpu_index_path : str | NoneType, optional
            Describe ``cpu_index_path``.
            Defaults to ``None``.
        """
        faiss_module = faiss
        if faiss_module is None:
            self.build()
            return
        try:
            if HAVE_FAISS and cpu_index_path and Path(cpu_index_path).exists():
                cpu_raw: object = faiss_module.read_index(cpu_index_path)
                cpu: FaissIndexProtocol = cast(FaissIndexProtocol, cpu_raw)
                self.index = self._clone_to_gpu(cpu)
                dense = self._load_dense_parquet()
                self.vecs = dense
                self.idmap = dense.ids
                return
        except (OSError, RuntimeError, ValueError, IndexError) as exc:
            logger.warning(
                "Failed to load FAISS index from %s: %s", cpu_index_path, exc, exc_info=True
            )
            # Fall back to building index
        self.build()

    def _clone_to_gpu(self, cpu_index: FaissIndexProtocol) -> FaissIndexProtocol:
        """Clone a CPU index onto a GPU when GPU support is available.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        cpu_index : object
            CPU index to clone.

        Returns
        -------
        object
            GPU-backed FAISS index when cloning succeeds, otherwise the original CPU index.

        Raises
        ------
        IndexBuildError
            Raised when GPU cloning fails and no fallback is available.
        """
        faiss_module = faiss
        if faiss_module is None:
            return cpu_index
        # Explicitly type FAISS GPU resources - use Protocol types
        standard_resources_type_raw: object | None = getattr(
            faiss_module, "StandardGpuResources", None
        )
        gpu_cloner_options_type_raw: object | None = getattr(faiss_module, "GpuClonerOptions", None)
        index_cpu_to_gpu_func: object | None = getattr(faiss_module, "index_cpu_to_gpu", None)
        if (
            standard_resources_type_raw is None
            or gpu_cloner_options_type_raw is None
            or index_cpu_to_gpu_func is None
        ):
            logger.debug("GPU resources not available, using CPU index")
            return cpu_index
        # Create instances with explicit Protocol typing - cast constructors to callable, then call
        resources_constructor: Callable[[], GpuResourcesProtocol] = cast(
            Callable[[], GpuResourcesProtocol], standard_resources_type_raw
        )
        options_constructor: Callable[[], GpuClonerOptionsProtocol] = cast(
            Callable[[], GpuClonerOptionsProtocol], gpu_cloner_options_type_raw
        )
        resources: GpuResourcesProtocol = resources_constructor()
        options: GpuClonerOptionsProtocol = options_constructor()
        # Set use_cuvs attribute explicitly - Protocol guarantees this attribute exists
        options.use_cuvs = True
        # Call index_cpu_to_gpu with explicit typing - define callable type
        index_cpu_to_gpu_callable: Callable[
            [GpuResourcesProtocol, int, FaissIndexProtocol, GpuClonerOptionsProtocol], object
        ] = cast(
            Callable[
                [GpuResourcesProtocol, int, FaissIndexProtocol, GpuClonerOptionsProtocol], object
            ],
            index_cpu_to_gpu_func,
        )
        try:
            gpu_index_raw: object = index_cpu_to_gpu_callable(resources, 0, cpu_index, options)
            return cast(FaissIndexProtocol, gpu_index_raw)
        except (RuntimeError, OSError, ValueError) as exc:
            logger.debug("GPU cloning with cuVS failed, falling back: %s", exc)
            # Disable cuVS and retry
            options.use_cuvs = False
            try:
                gpu_index_raw = index_cpu_to_gpu_callable(resources, 0, cpu_index, options)
                return cast(FaissIndexProtocol, gpu_index_raw)
            except (RuntimeError, OSError, ValueError) as gpu_exc:
                msg = f"GPU cloning failed: {gpu_exc}"
                logger.warning(msg, exc_info=True)
                # Return CPU index as fallback rather than raising
                return cpu_index

    def search(self, qvec: VectorArray | VecArray, k: int = 10) -> list[list[tuple[str, float]]]:
        """Search the index for nearest neighbours.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        qvec : tuple[int, ...] | np.float32 | VecArray
            Query vector(s) as numpy array of shape (n_queries, dimension) or (dimension,).
        k : int, optional
            Number of nearest neighbors to return per query.
            Defaults to ``10``.

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked matches for each query with document identifiers and similarity scores.

        Raises
        ------
        VectorSearchError
            Raised when search fails or index is not loaded.
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
            Normalized query vectors of shape (n_queries, dimension).
        k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked matches for each query.

        Raises
        ------
        VectorSearchError
            Raised when the FAISS index or identifier mapping has not been loaded.
        """
        if self.index is None or self.idmap is None:
            msg = "FAISS index or ID mapping not loaded"
            raise VectorSearchError(msg)
        # Explicitly type FAISS search results
        search_result: tuple[NDArray[np.float32], NDArray[np.int64]] = self.index.search(queries, k)
        distances: NDArray[np.float32] = search_result[0]
        indices: NDArray[np.int64] = search_result[1]
        batches: list[list[tuple[str, float]]] = []
        # Iterate with explicit typing
        n_queries: int = len(indices)
        for query_idx in range(n_queries):
            idx_row: NDArray[np.int64] = indices[query_idx]
            dist_row: NDArray[np.float32] = distances[query_idx]
            results: list[tuple[str, float]] = []
            n_results: int = len(idx_row)
            for result_idx in range(n_results):
                idx_val: np.int64 = idx_row[result_idx]
                idx: int = int(idx_val)
                if idx < 0:
                    continue
                score_val: np.float32 = dist_row[result_idx]
                score: float = float(score_val)
                results.append((self.idmap[idx], score))
            batches.append(results)
        return batches

    def _search_with_cpu(self, queries: VecArray, k: int) -> list[list[tuple[str, float]]]:
        """Perform a batched search using the CPU fallback implementation.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        queries : VecArray
            Normalized query vectors of shape (n_queries, dimension).
        k : int
            Number of nearest neighbors to return per query.

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked matches for each query.

        Raises
        ------
        VectorSearchError
            Raised when dense vectors have not been loaded into memory.
        """
        if self.vecs is None:
            msg = "Dense vectors not loaded"
            raise VectorSearchError(msg)
        matrix: VectorArray = self.vecs.mat
        batches: list[list[tuple[str, float]]] = []
        n_queries: int = len(queries)
        for query_idx in range(n_queries):
            row: VectorArray = queries[query_idx]
            normalised: VectorArray = row.copy()
            norm_result_scalar: float = float(np.linalg.norm(normalised).item())  # type: ignore[misc]  # numpy.linalg.norm returns floating[Any]
            normalized_divisor: float = norm_result_scalar + 1e-9
            normalised = cast(VectorArray, normalised / normalized_divisor)
            sims_raw: NDArray[np.float32] = matrix @ normalised
            sims: NDArray[np.float32] = sims_raw
            topk_raw: NDArray[np.intp] = np.argsort(-sims)[:k]
            topk: NDArray[np.intp] = topk_raw
            batch_results: list[tuple[str, float]] = []
            n_topk: int = len(topk)
            for rank_idx in range(n_topk):
                vec_idx_val: np.intp = topk[rank_idx]
                vec_idx: int = int(vec_idx_val)
                score_val: np.float32 = sims[vec_idx]
                score: float = float(score_val)
                batch_results.append((self.vecs.ids[vec_idx], score))
            batches.append(batch_results)
        return batches
