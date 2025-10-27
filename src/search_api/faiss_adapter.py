"""Module for search_api.faiss_adapter.

NavMap:
- DenseVecs: Densevecs.
- FaissAdapter: Faissadapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

try:
    try:
        from libcuvs import load_library as _load_cuvs

        _load_cuvs()
    except Exception:
        pass
    import faiss

    HAVE_FAISS = True
except Exception:
    faiss = None
    HAVE_FAISS = False


@dataclass
class DenseVecs:
    """Densevecs."""

    ids: list[str]
    mat: np.ndarray


class FaissAdapter:
    """Faissadapter."""

    def __init__(
        self, db_path: str, factory: str = "OPQ64,IVF8192,PQ64", metric: str = "ip"
    ) -> None:
        """Init.

        Parameters
        ----------
        db_path : str
            TODO.
        factory : str
            TODO.
        metric : str
            TODO.
        """
        self.db_path = db_path
        self.factory = factory
        self.metric = metric
        self.index: Any | None = None
        self.idmap: list[str] | None = None
        self.vecs: DenseVecs | None = None

    def _load_dense_parquet(self) -> DenseVecs:
        """Load dense parquet.

        Returns
        -------
        DenseVecs
            TODO.
        """
        if not Path(self.db_path).exists():
            message = "DuckDB registry not found"
            raise RuntimeError(message)
        con = duckdb.connect(self.db_path)
        try:
            dr = con.execute(
                "SELECT parquet_root, dim FROM dense_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not dr:
                message = "No dense_runs found"
                raise RuntimeError(message)
            root = dr[0]
            rows = con.execute(
                f"""
              SELECT chunk_id, vector FROM read_parquet('{root}/*/*.parquet', union_by_name=true)
            """
            ).fetchall()
        finally:
            con.close()
        ids = [r[0] for r in rows]
        mat = np.stack([np.array(r[1], dtype=np.float32) for r in rows])
        # normalize for cosine/IP
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        return DenseVecs(ids=ids, mat=mat)

    def build(self) -> None:
        """Build.

        Returns
        -------
        None
            TODO.
        """
        vecs = self._load_dense_parquet()
        self.vecs = vecs
        if not HAVE_FAISS:
            return
        d = vecs.mat.shape[1]
        metric = faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
        cpu = faiss.index_factory(d, self.factory, metric)
        cpu = faiss.IndexIDMap2(cpu)
        train = vecs.mat[: min(100000, vecs.mat.shape[0])].copy()
        faiss.normalize_L2(train)
        cpu.train(train)
        ids64 = np.arange(vecs.mat.shape[0], dtype=np.int64)
        cpu.add_with_ids(vecs.mat, ids64)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.use_cuvs = True
        try:
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu, co)
        except Exception:
            co.use_cuvs = False
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu, co)
        self.idmap = vecs.ids

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Load or build.

        Parameters
        ----------
        cpu_index_path : Optional[str]
            TODO.

        Returns
        -------
        None
            TODO.
        """
        try:
            if HAVE_FAISS and cpu_index_path and Path(cpu_index_path).exists():
                cpu = faiss.read_index(cpu_index_path)
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.use_cuvs = True
                try:
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu, co)
                except Exception:
                    co.use_cuvs = False
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu, co)
                dv = self._load_dense_parquet()
                self.vecs = dv
                self.idmap = dv.ids
                return
        except Exception:
            pass
        self.build()

    def search(self, qvec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Search.

        Parameters
        ----------
        qvec : np.ndarray
            TODO.
        k : int
            TODO.

        Returns
        -------
        List[Tuple[str, float]]
            TODO.
        """
        if self.vecs is None and self.index is None:
            return []
        if HAVE_FAISS and self.index is not None:
            if self.idmap is None:
                message = "ID mapping not loaded for FAISS index"
                raise RuntimeError(message)
            q = qvec[None, :].astype(np.float32, copy=False)
            distances, indices = self.index.search(q, k)
            out = []
            for idx, score in zip(indices[0], distances[0], strict=False):
                if idx < 0:
                    continue
                out.append((self.idmap[int(idx)], float(score)))
            return out
        # numpy brute force
        if self.vecs is None:
            message = "Dense vectors not loaded"
            raise RuntimeError(message)
        mat = self.vecs.mat
        q = qvec.astype(np.float32, copy=False)
        q /= np.linalg.norm(q) + 1e-9
        sims = mat @ q
        topk = np.argsort(-sims)[:k]
        ids = self.vecs.ids
        return [(ids[i], float(sims[i])) for i in topk]
