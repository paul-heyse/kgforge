"""Gpu utilities."""

from __future__ import annotations

import os
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap

__all__ = ["FaissGpuIndex"]

__navmap__: Final[NavMap] = {
    "title": "vectorstore_faiss.gpu",
    "synopsis": "FAISS index wrapper with optional cuVS acceleration and numpy fallback.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["FaissGpuIndex"],
        },
    ],
}

type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]
type StrArray = NDArray[np.str_]


# [nav:anchor FaissGpuIndex]
class FaissGpuIndex:
    """Describe FaissGpuIndex."""

    def __init__(
        self,
        factory: str = "OPQ64,IVF8192,PQ64",
        nprobe: int = 64,
        gpu: bool = True,
        cuvs: bool = True,
    ) -> None:
        """Compute init.

        Initialise a new instance with validated parameters.

        Parameters
        ----------
        factory : str | None
            Description for ``factory``.
        nprobe : int | None
            Description for ``nprobe``.
        gpu : bool | None
            Description for ``gpu``.
        cuvs : bool | None
            Description for ``cuvs``.
        """
        
        
        
        
        
        
        
        
        
        self.factory = factory
        self.nprobe = nprobe
        self.gpu = gpu
        self.cuvs = cuvs
        self._faiss: Any | None = None
        self._res: Any | None = None
        self._index: Any | None = None
        self._idmap: StrArray | None = None
        self._xb: FloatArray | None = None
        try:
            import faiss

            self._faiss = faiss
        except Exception:  # pragma: no cover - environment without FAISS
            self._faiss = None

    def _ensure_resources(self) -> None:
        """Compute ensure resources.

        Carry out the ensure resources operation.
        """
        if not self._faiss or not self.gpu:
            return
        if self._res is None:
            faiss = self._faiss
            self._res = faiss.StandardGpuResources()

    def train(self, train_vectors: FloatArray, *, seed: int = 42) -> None:
        """Compute train.

        Carry out the train operation.

        Parameters
        ----------
        train_vectors : src.vectorstore_faiss.gpu.FloatArray
            Description for ``train_vectors``.
        seed : int | None
            Description for ``seed``.
        """
        
        
        
        
        
        
        
        
        
        if self._faiss is None:
            return
        train_mat = cast(FloatArray, np.asarray(train_vectors, dtype=np.float32, order="C"))
        faiss = self._faiss
        dimension = train_mat.shape[1]
        cpu_index = faiss.index_factory(dimension, self.factory, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(train_mat)
        cpu_index.train(train_mat)
        self._ensure_resources()
        if self.gpu:
            options = faiss.GpuClonerOptions()
            if hasattr(options, "use_cuvs"):
                options.use_cuvs = bool(self.cuvs)
            try:
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index, options)
            except Exception:  # pragma: no cover - fallback without cuVS
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index)
        else:
            self._index = cpu_index
        try:
            params = faiss.GpuParameterSpace() if self.gpu else faiss.ParameterSpace()
            params.set_index_parameter(self._index, "nprobe", self.nprobe)
        except Exception:
            pass

    def add(self, keys: list[str], vectors: FloatArray) -> None:
        """Compute add.

        Carry out the add operation.

        Parameters
        ----------
        keys : List[str]
            Description for ``keys``.
        vectors : src.vectorstore_faiss.gpu.FloatArray
            Description for ``vectors``.

        Raises
        ------
        RuntimeError
            Raised when validation fails.
        """
        
        
        
        
        
        
        
        
        
        vec_array = cast(FloatArray, np.asarray(vectors, dtype=np.float32, order="C"))
        if self._faiss is None:
            self._xb = cast(FloatArray, np.array(vec_array, copy=True))
            self._idmap = cast(StrArray, np.asarray(keys, dtype=str))
            norms = np.linalg.norm(self._xb, axis=1, keepdims=True) + 1e-12
            self._xb /= norms
            return
        faiss = self._faiss
        if self._index is None:
            message = "FAISS index not initialized; call train() before add()."
            raise RuntimeError(message)
        faiss.normalize_L2(vec_array)
        if isinstance(self._index, faiss.IndexIDMap2):
            idmap_array = cast(IntArray, np.asarray(keys, dtype="int64"))
            self._index.add_with_ids(vec_array, idmap_array)
        elif hasattr(faiss, "IndexIDMap2"):
            idmap = faiss.IndexIDMap2(self._index)
            idmap_array = cast(IntArray, np.asarray(keys, dtype="int64"))
            idmap.add_with_ids(vec_array, idmap_array)
            self._index = idmap
        else:
            self._index.add(vec_array)

    def search(self, query: FloatArray, k: int) -> list[tuple[str, float]]:
        """Compute search.

        Carry out the search operation.

        Parameters
        ----------
        query : src.vectorstore_faiss.gpu.FloatArray
            Description for ``query``.
        k : int
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
        
        
        
        
        
        
        
        
        
        q = cast(FloatArray, np.asarray(query, dtype=np.float32, order="C"))
        q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
        if self._faiss is None or self._index is None:
            if self._xb is None or self._idmap is None:
                return []
            sims_matrix = self._xb @ q.T
            sims = np.asarray(sims_matrix, dtype=np.float32).squeeze()
            indices = np.argsort(-sims)[:k]
            return [(str(self._idmap[i]), float(sims[i])) for i in indices.tolist()]
        if self._idmap is None:
            message = "ID map not loaded; cannot resolve FAISS results."
            raise RuntimeError(message)
        distances, indices = self._index.search(q.reshape(1, -1), k)
        ids = indices[0]
        scores = distances[0]
        return [(str(ids[i]), float(scores[i])) for i in range(len(ids)) if ids[i] != -1]

    def save(self, index_uri: str, idmap_uri: str | None = None) -> None:
        """Compute save.

        Carry out the save operation.

        Parameters
        ----------
        index_uri : str
            Description for ``index_uri``.
        idmap_uri : str | None
            Description for ``idmap_uri``.

        Raises
        ------
        RuntimeError
            Raised when validation fails.
        """
        
        
        
        
        
        
        
        
        
        if self._faiss is None or self._index is None:
            if self._xb is not None and self._idmap is not None:
                np.savez(index_uri, xb=self._xb, ids=self._idmap)
            return
        faiss = self._faiss
        if faiss is None:
            message = "FAISS not available"
            raise RuntimeError(message)
        target_index = faiss.index_gpu_to_cpu(self._index) if self.gpu else self._index
        faiss.write_index(target_index, index_uri)

    def load(self, index_uri: str, idmap_uri: str | None = None) -> None:
        """Compute load.

        Carry out the load operation.

        Parameters
        ----------
        index_uri : str
            Description for ``index_uri``.
        idmap_uri : str | None
            Description for ``idmap_uri``.

        Raises
        ------
        RuntimeError
            Raised when validation fails.
        """
        
        
        
        
        
        
        
        
        
        if self._faiss is None:
            if os.path.exists(index_uri + ".npz"):
                data = np.load(index_uri + ".npz", allow_pickle=True)
                self._xb = data["xb"]
                self._idmap = data["ids"]
            return
        faiss = self._faiss
        if faiss is None:
            message = "FAISS not available"
            raise RuntimeError(message)
        cpu_index = faiss.read_index(index_uri)
        self._ensure_resources()
        if self.gpu:
            options = faiss.GpuClonerOptions()
            if hasattr(options, "use_cuvs"):
                options.use_cuvs = bool(self.cuvs)
            try:
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index, options)
            except Exception:
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index)
        else:
            self._index = cpu_index
