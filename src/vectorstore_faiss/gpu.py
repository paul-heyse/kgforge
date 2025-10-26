
from __future__ import annotations
import os
from typing import List, Tuple, Optional
import numpy as np

class FaissGpuIndex:
    \"""FAISS GPU/cuVS wrapper.
    If FAISS is unavailable at runtime, falls back to brute-force cosine search over a provided matrix.\"""
    def __init__(self, factory: str = "OPQ64,IVF8192,PQ64", nprobe: int = 64, gpu: bool = True, cuvs: bool = True):
        self.factory = factory
        self.nprobe = nprobe
        self.gpu = gpu
        self.cuvs = cuvs
        self._faiss = None
        self._res = None
        self._index = None
        self._idmap: Optional[np.ndarray] = None
        self._xb: Optional[np.ndarray] = None  # fallback matrix for brute force
        # try import faiss
        try:
            import faiss
            self._faiss = faiss
        except Exception:
            self._faiss = None

    def _ensure_resources(self):
        if not self._faiss or not self.gpu:
            return
        if self._res is None:
            faiss = self._faiss
            self._res = faiss.StandardGpuResources()
            # memory knobs can be tuned by caller later

    def train(self, train_vectors: np.ndarray, *, seed: int = 42) -> None:
        if self._faiss is None:
            # no-op in fallback; brute force doesn't need training
            return
        faiss = self._faiss
        d = train_vectors.shape[1]
        cpu_index = faiss.index_factory(d, self.factory, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(train_vectors)
        cpu_index.train(train_vectors)
        self._ensure_resources()
        if self.gpu:
            co = faiss.GpuClonerOptions()
            if hasattr(co, "use_cuvs"):
                co.use_cuvs = bool(self.cuvs)
            try:
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index, co)
            except Exception:
                # fallback without cuVS
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index)
        else:
            self._index = cpu_index
        # set nprobe
        try:
            ps = faiss.GpuParameterSpace() if self.gpu else faiss.ParameterSpace()
            ps.set_index_parameter(self._index, "nprobe", self.nprobe)
        except Exception:
            pass

    def add(self, keys: List[str], vectors: np.ndarray) -> None:
        if self._faiss is None:
            # fallback: keep matrix and id map for brute-force
            self._xb = vectors.astype("float32", copy=True)
            self._idmap = np.array(keys)
            # normalize for cosine/IP
            norms = np.linalg.norm(self._xb, axis=1, keepdims=True) + 1e-12
            self._xb /= norms
            return
        faiss = self._faiss
        faiss.normalize_L2(vectors)
        if isinstance(self._index, faiss.IndexIDMap2):
            self._index.add_with_ids(vectors, np.array(keys, dtype="int64"))
        elif hasattr(faiss, "IndexIDMap2"):
            # wrap once
            idmap = faiss.IndexIDMap2(self._index)
            idmap.add_with_ids(vectors, np.array(keys, dtype="int64"))
            self._index = idmap
        else:
            self._index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        q = query.astype("float32", copy=True)
        # normalize for cosine/IP
        q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        if self._faiss is None or self._index is None:
            # brute-force cosine over self._xb
            if self._xb is None or self._idmap is None:
                return []
            sims = (self._xb @ q.T).squeeze()
            idx = np.argsort(-sims)[:k]
            return [(str(self._idmap[i]), float(sims[i])) for i in idx.tolist()]
        faiss = self._faiss
        D, I = self._index.search(q.reshape(1, -1), k)
        # map IDs to strings if using IDMap; else cast ints
        ids = I[0]
        scores = D[0]
        return [(str(ids[i]), float(scores[i])) for i in range(len(ids)) if ids[i] != -1]

    def save(self, index_uri: str, idmap_uri: str) -> None:
        if self._faiss is None or self._index is None:
            # save fallback matrix
            if self._xb is not None and self._idmap is not None:
                np.savez(index_uri, xb=self._xb, ids=self._idmap)
            return
        faiss = self._faiss
        faiss.write_index(faiss.index_gpu_to_cpu(self._index) if self.gpu else self._index, index_uri)

    def load(self, index_uri: str, idmap_uri: str | None = None) -> None:
        if self._faiss is None:
            # load fallback matrix
            if os.path.exists(index_uri + ".npz"):
                data = np.load(index_uri + ".npz", allow_pickle=True)
                self._xb = data["xb"]
                self._idmap = data["ids"]
            return
        faiss = self._faiss
        cpu_index = faiss.read_index(index_uri)
        self._ensure_resources()
        if self.gpu:
            co = faiss.GpuClonerOptions()
            if hasattr(co, "use_cuvs"):
                co.use_cuvs = bool(self.cuvs)
            try:
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index, co)
            except Exception:
                self._index = faiss.index_cpu_to_gpu(self._res, 0, cpu_index)
        else:
            self._index = cpu_index
