"""Overview of gpu.

This module bundles gpu logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import os
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap

__all__ = ["FaissGpuIndex", "FloatArray", "IntArray", "StrArray"]

__navmap__: Final[NavMap] = {
    "title": "vectorstore_faiss.gpu",
    "synopsis": "GPU-accelerated FAISS bindings and helper types",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}

# [nav:anchor FloatArray]
type FloatArray = NDArray[np.float32]
type FloatArrayLike = NDArray[np.floating[Any]]

# [nav:anchor IntArray]
type IntArray = NDArray[np.int64]

# [nav:anchor StrArray]
type StrArray = NDArray[np.str_]


# [nav:anchor FaissGpuIndex]
class FaissGpuIndex:
    """Describe FaissGpuIndex.

    <!-- auto:docstring-builder v1 -->

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.

    Parameters
    ----------
    factory : str, optional
        Describe ``factory``.
        Defaults to ``'OPQ64,IVF8192,PQ64'``.
    nprobe : int, optional
        Describe ``nprobe``.
        Defaults to ``64``.
    gpu : bool, optional
        Describe ``gpu``.
        Defaults to ``True``.
    cuvs : bool, optional
        Describe ``cuvs``.
        Defaults to ``True``.

    Raises
    ------
    RuntimeError
    Raised when TODO for RuntimeError.
    """

    def __init__(
        self,
        factory: str = "OPQ64,IVF8192,PQ64",
        nprobe: int = 64,
        gpu: bool = True,
        cuvs: bool = True,
    ) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        factory : str, optional
            Describe ``factory``.
            Defaults to ``'OPQ64,IVF8192,PQ64'``.
        nprobe : int, optional
            Describe ``nprobe``.
            Defaults to ``64``.
        gpu : bool, optional
            Describe ``gpu``.
            Defaults to ``True``.
        cuvs : bool, optional
            Describe ``cuvs``.
            Defaults to ``True``.
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
        """Describe  ensure resources.

        <!-- auto:docstring-builder v1 -->

        Python's object protocol for this class. Use it to integrate with built-in operators,
        protocols, or runtime behaviours that expect instances to participate in the language's data
        model.
        """
        if not self._faiss or not self.gpu:
            return
        if self._res is None:
            faiss = self._faiss
            self._res = faiss.StandardGpuResources()

    def train(self, train_vectors: FloatArray | FloatArrayLike, *, seed: int = 42) -> None:
        """Describe train.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        train_vectors : FloatArray | FloatArrayLike
            Describe ``train_vectors``.
        seed : int, optional
            Describe ``seed``.
            Defaults to ``42``.
        """
        if self._faiss is None:
            return
        train_mat: FloatArray = np.asarray(train_vectors, dtype=np.float32, order="C")
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

    def add(self, keys: list[str], vectors: FloatArray | FloatArrayLike) -> None:
        """Describe add.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        keys : list[str]
            Describe ``keys``.
        vectors : FloatArray | FloatArrayLike
            Describe ``vectors``.

        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
        """
        vec_array: FloatArray = np.asarray(vectors, dtype=np.float32, order="C")
        if self._faiss is None:
            xb: FloatArray = np.array(vec_array, dtype=np.float32, copy=True)
            self._xb = xb
            idmap: StrArray = np.asarray(keys, dtype=np.str_)
            self._idmap = idmap
            norms = np.linalg.norm(self._xb, axis=1, keepdims=True) + 1e-12
            self._xb /= norms
            return
        faiss = self._faiss
        if self._index is None:
            message = "FAISS index not initialized; call train() before add()."
            raise RuntimeError(message)
        faiss.normalize_L2(vec_array)
        idmap_array: IntArray = np.asarray(keys, dtype=np.int64)
        if isinstance(self._index, faiss.IndexIDMap2):
            self._index.add_with_ids(vec_array, idmap_array)
        elif hasattr(faiss, "IndexIDMap2"):
            idmap = faiss.IndexIDMap2(self._index)
            idmap.add_with_ids(vec_array, idmap_array)
            self._index = idmap
        else:
            self._index.add(vec_array)

    def search(self, query: FloatArray | FloatArrayLike, k: int) -> list[tuple[str, float]]:
        """Describe search.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        query : FloatArray | FloatArrayLike
            Describe ``query``.
        k : int
            Describe ``k``.

        Returns
        -------
        list[tuple[str, float]]
            Describe return value.










        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
        """
        q: FloatArray = np.asarray(query, dtype=np.float32, order="C")
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
        """Describe save.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index_uri : str
            Describe ``index_uri``.
        idmap_uri : str | None, optional
            Describe ``idmap_uri``.
            Defaults to ``None``.

        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
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
        """Describe load.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index_uri : str
            Describe ``index_uri``.
        idmap_uri : str | None, optional
            Describe ``idmap_uri``.
            Defaults to ``None``.

        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
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
