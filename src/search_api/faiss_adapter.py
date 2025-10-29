"""Overview of faiss adapter.

This module bundles faiss adapter logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


# [nav:anchor DenseVecs]
@dataclass
class DenseVecs:
    """Dense vector matrix and identifiers loaded from storage.
<!-- auto:docstring-builder v1 -->

    Attributes
    ----------
    ids : list[str]
        Chunk identifiers aligned with ``mat`` rows.
    mat : VecArray
        Normalised dense vectors suitable for cosine similarity.
    """

    ids: list[str]
    mat: VecArray


# [nav:anchor FaissAdapter]
class FaissAdapter:
    """Dense retrieval adapter that couples DuckDB storage with FAISS.
<!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database or parquet directory containing vectors.
    factory : str, optional
        FAISS index factory string used when building GPU/CPU indices.
        Defaults to ``"OPQ64,IVF8192,PQ64"``.
    metric : str, optional
        Similarity metric used for search, e.g. ``"ip"`` or ``"l2"``. Defaults
        to ``"ip"``.
    """

    def __init__(
        self,
        db_path: str,
        factory: str = "OPQ64,IVF8192,PQ64",
        metric: str = "ip",
    ) -> None:
        self.db_path = db_path
        self.factory = factory
        self.metric = metric
        self.index: Any | None = None
        self.idmap: list[str] | None = None
        self.vecs: DenseVecs | None = None

    def _load_dense_from_parquet(self, source: Path) -> DenseVecs:
        """Load and normalise dense vectors from a parquet dataset.
<!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        source : Path
            Path to a parquet file or directory readable by DuckDB.

        Returns
        -------
        DenseVecs
            Identifiers and normalised vectors backed by ``float32`` arrays.

        Raises
        ------
        duckdb.Error
            If DuckDB fails to read the parquet source.
        RuntimeError
            If no vectors are discovered in ``source``.
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
        """Compute load dense parquet.
<!-- auto:docstring-builder v1 -->

Carry out the load dense parquet operation.

Returns
-------
DenseVecs
    Description of return value.
    
    
    

Raises
------
RuntimeError
    Raised when validation fails.
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
        """Compute build.
<!-- auto:docstring-builder v1 -->

Carry out the build operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Examples
--------
>>> from search_api.faiss_adapter import build
>>> build()  # doctest: +ELLIPSIS
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
        cpu.train(n=train.shape[0], x=train)
        ids64 = cast(IndexArray, np.arange(vectors.mat.shape[0], dtype=np.int64))
        cpu.add_with_ids(n=vectors.mat.shape[0], x=vectors.mat, xids=ids64)
        self.index = self._clone_to_gpu(cpu)

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Compute load or build.
<!-- auto:docstring-builder v1 -->

Carry out the load or build operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
cpu_index_path : str | None, optional
    Defaults to ``None``.
    Description for ``cpu_index_path``.
    
    
    
    Defaults to ``None``.

Examples
--------
>>> from search_api.faiss_adapter import load_or_build
>>> load_or_build()  # doctest: +ELLIPSIS
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
        """Return a GPU-backed index when FAISS provides the necessary bindings.
<!-- auto:docstring-builder v1 -->

Parameters
----------
cpu_index : object
    Describe ``cpu_index``.
    
    

Returns
-------
object
    Describe return value.
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

    def search(self, qvec: VecArray, k: int = 10) -> list[list[tuple[str, float]]]:
        """Compute search.
<!-- auto:docstring-builder v1 -->

Carry out the search operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
qvec : VecArray
    Description for ``qvec``.
k : int, optional
    Defaults to ``10``.
    Description for ``k``.
    
    
    
    Defaults to ``10``.

Returns
-------
list[list[tuple[str, float]]]
    Description of return value.
    
    
    

Examples
--------
>>> from search_api.faiss_adapter import search
>>> result = search(...)
>>> result  # doctest: +ELLIPSIS
"""
        if self.vecs is None and self.index is None:
            return []
        queries = self._prepare_queries(qvec)
        if HAVE_FAISS and self.index is not None:
            return self._search_with_faiss(queries, k)
        return self._search_with_cpu(queries, k)

    def _prepare_queries(self, qvec: VecArray) -> VecArray:
        """Normalise query input into a 2D float32 array.
<!-- auto:docstring-builder v1 -->

Parameters
----------
qvec : VecArray
    Describe ``qvec``.
    
    

Returns
-------
VecArray
    Describe return value.
"""
        query_arr: VecArray = np.asarray(qvec, dtype=np.float32, order="C")
        if query_arr.ndim == 1:
            query_arr = query_arr[None, :]
        return query_arr

    def _search_with_faiss(self, queries: VecArray, k: int) -> list[list[tuple[str, float]]]:
        """Execute a FAISS search using the configured index.
<!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        queries : VecArray
            2-D array of query vectors.
        k : int
            Number of nearest neighbours to return for each query.

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked chunk identifiers paired with similarity scores.

        Raises
        ------
        RuntimeError
            If the FAISS index or identifier map has not been loaded.
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
        """Perform cosine similarity search using the in-memory matrix.
<!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        queries : VecArray
            2-D array of query vectors.
        k : int
            Number of nearest neighbours to return for each query.

        Returns
        -------
        list[list[tuple[str, float]]]
            Ranked chunk identifiers paired with similarity scores.

        Raises
        ------
        RuntimeError
            If dense vectors have not been loaded into memory.
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
