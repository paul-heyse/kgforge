"""Faiss Adapter utilities."""

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
            "symbols": ["DenseVecs", "FaissAdapter"],
        },
    ],
}

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


type VecArray = NDArray[np.float32]
type IndexArray = NDArray[np.int64]


# [nav:anchor DenseVecs]
@dataclass
class DenseVecs:
    """Describe DenseVecs."""

    ids: list[str]
    mat: VecArray


# [nav:anchor FaissAdapter]
class FaissAdapter:
    """Describe FaissAdapter."""

    def __init__(
        self,
        db_path: str,
        factory: str = "OPQ64,IVF8192,PQ64",
        metric: str = "ip",
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        db_path : str
            Description for ``db_path``.
        factory : str | None
            Description for ``factory``.
        metric : str | None
            Description for ``metric``.
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
















        self.db_path = db_path
        self.factory = factory
        self.metric = metric
        self.index: Any | None = None
        self.idmap: list[str] | None = None
        self.vecs: DenseVecs | None = None

    def _load_dense_parquet(self) -> DenseVecs:
        """Compute load dense parquet.

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
        if not Path(self.db_path).exists():
            message = "DuckDB registry not found"
            raise RuntimeError(message)
        con = duckdb.connect(self.db_path)
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
        mat = cast(VecArray, np.stack([np.asarray(row[1], dtype=np.float32) for row in rows]))
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        normalized = cast(VecArray, mat / norms)
        return DenseVecs(ids=ids, mat=normalized)

    def build(self) -> None:
        """Compute build.

        Carry out the build operation.
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
















        vectors = self._load_dense_parquet()
        self.vecs = vectors
        if not HAVE_FAISS:
            return
        dimension = vectors.mat.shape[1]
        metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
        cpu = faiss.index_factory(dimension, self.factory, metric_type)
        cpu = faiss.IndexIDMap2(cpu)
        train = vectors.mat[: min(100000, vectors.mat.shape[0])].copy()
        faiss.normalize_L2(train)
        cpu.train(train)
        ids64 = cast(IndexArray, np.arange(vectors.mat.shape[0], dtype=np.int64))
        cpu.add_with_ids(vectors.mat, ids64)
        resources = faiss.StandardGpuResources()
        options = faiss.GpuClonerOptions()
        options.use_cuvs = True
        try:
            self.index = faiss.index_cpu_to_gpu(resources, 0, cpu, options)
        except Exception:  # pragma: no cover - fallback path without cuVS
            options.use_cuvs = False
            self.index = faiss.index_cpu_to_gpu(resources, 0, cpu, options)
        self.idmap = vectors.ids

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Compute load or build.

        Carry out the load or build operation.

        Parameters
        ----------
        cpu_index_path : str | None
            Description for ``cpu_index_path``.
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
















        try:
            if HAVE_FAISS and cpu_index_path and Path(cpu_index_path).exists():
                cpu = faiss.read_index(cpu_index_path)
                resources = faiss.StandardGpuResources()
                options = faiss.GpuClonerOptions()
                options.use_cuvs = True
                try:
                    self.index = faiss.index_cpu_to_gpu(resources, 0, cpu, options)
                except Exception:  # pragma: no cover - fallback path without cuVS
                    options.use_cuvs = False
                    self.index = faiss.index_cpu_to_gpu(resources, 0, cpu, options)
                dense = self._load_dense_parquet()
                self.vecs = dense
                self.idmap = dense.ids
                return
        except Exception:
            pass
        self.build()

    def search(self, qvec: VecArray, k: int = 10) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation.

        Parameters
        ----------
        qvec : src.search_api.faiss_adapter.VecArray
            Description for ``qvec``.
        k : int | None
            Description for ``k``.

        Returns
        -------
        List[Tuple[str, float]]
            Description of return value.

        Raises
        ------
        RuntimeError
            Raised when validation fails.
        """
        
        
        
        
        
        
        
        
        
        
        
        
        
















        if self.vecs is None and self.index is None:
            return []
        if HAVE_FAISS and self.index is not None:
            if self.idmap is None:
                message = "ID mapping not loaded for FAISS index"
                raise RuntimeError(message)
            query = cast(VecArray, np.asarray(qvec[None, :], dtype=np.float32, order="C"))
            distances, indices = self.index.search(query, k)
            results: list[tuple[str, float]] = []
            for idx, score in zip(indices[0], distances[0], strict=False):
                if idx < 0:
                    continue
                results.append((self.idmap[int(idx)], float(score)))
            return results
        if self.vecs is None:
            message = "Dense vectors not loaded"
            raise RuntimeError(message)
        matrix = self.vecs.mat
        query = cast(VecArray, np.asarray(qvec, dtype=np.float32, order="C"))
        query /= np.linalg.norm(query) + 1e-9
        sims = matrix @ query
        topk = np.argsort(-sims)[:k]
        ids = self.vecs.ids
        return [(ids[i], float(sims[i])) for i in topk]
