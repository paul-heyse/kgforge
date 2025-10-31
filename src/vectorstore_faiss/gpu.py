"""Overview of gpu.

This module bundles gpu logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.navmap_types import NavMap

logger = logging.getLogger(__name__)

# Try importing FAISS at module level - handle gracefully if unavailable
try:
    import faiss
except (ImportError, ModuleNotFoundError):
    faiss = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from faiss import IndexIDMap2 as FaissIndexIDMap2

    from search_api.types import (
        FaissIndexProtocol,
        FaissModuleProtocol,
        GpuClonerOptionsProtocol,
        GpuResourcesProtocol,
    )
else:  # pragma: no cover - faiss may be missing at runtime
    FaissIndexIDMap2 = Any
    FaissIndexProtocol = object  # type: ignore[assignment, misc]
    FaissModuleProtocol = object  # type: ignore[assignment, misc]
    GpuClonerOptionsProtocol = object  # type: ignore[assignment, misc]
    GpuResourcesProtocol = object  # type: ignore[assignment, misc]

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
        self._faiss: FaissModuleProtocol | None = None
        self._res: GpuResourcesProtocol | None = None
        self._index: FaissIndexProtocol | None = None
        self._idmap: StrArray | None = None
        self._xb: FloatArray | None = None
        # FAISS module-level import handles availability check
        if faiss is not None:
            self._faiss = cast(FaissModuleProtocol, faiss)
        # else: self._faiss remains None from type annotation

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
            faiss_module: FaissModuleProtocol = self._faiss
            # StandardGpuResources constructor - cast to Protocol
            resources_constructor_raw: object = getattr(faiss_module, "StandardGpuResources", None)
            if resources_constructor_raw is not None:
                resources_constructor: Callable[[], GpuResourcesProtocol] = cast(
                    Callable[[], GpuResourcesProtocol], resources_constructor_raw
                )
                self._res = resources_constructor()

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
        del seed
        if self._faiss is None:
            return
        train_mat: FloatArray = np.asarray(train_vectors, dtype=np.float32, order="C")
        faiss_module: FaissModuleProtocol = self._faiss
        dimension = train_mat.shape[1]
        cpu_index_raw: object = faiss_module.index_factory(
            dimension, self.factory, faiss_module.METRIC_INNER_PRODUCT
        )
        cpu_index: FaissIndexProtocol = cast(FaissIndexProtocol, cpu_index_raw)
        faiss_module.normalize_L2(train_mat)
        cpu_index.train(train_mat)
        self._ensure_resources()
        if self.gpu:
            options_raw: object = getattr(faiss_module, "GpuClonerOptions", None)
            if options_raw is None:
                self._index = cpu_index
                return
            options_constructor: Callable[[], GpuClonerOptionsProtocol] = cast(
                Callable[[], GpuClonerOptionsProtocol], options_raw
            )
            options: GpuClonerOptionsProtocol = options_constructor()
            options.use_cuvs = bool(self.cuvs)
            try:
                index_cpu_to_gpu_func_raw: object = getattr(faiss_module, "index_cpu_to_gpu", None)
                if index_cpu_to_gpu_func_raw is None:
                    self._index = cpu_index
                    return
                index_cpu_to_gpu_callable: Callable[
                    [GpuResourcesProtocol, int, FaissIndexProtocol, GpuClonerOptionsProtocol],
                    object,
                ] = cast(
                    Callable[
                        [GpuResourcesProtocol, int, FaissIndexProtocol, GpuClonerOptionsProtocol],
                        object,
                    ],
                    index_cpu_to_gpu_func_raw,
                )
                if self._res is not None:
                    gpu_index_raw: object = index_cpu_to_gpu_callable(
                        self._res, 0, cpu_index, options
                    )
                    self._index = cast(FaissIndexProtocol, gpu_index_raw)
                else:
                    self._index = cpu_index
            except (RuntimeError, OSError, ValueError) as exc:
                logger.debug("GPU cloning with cuVS failed, falling back to standard GPU: %s", exc)
                if self._res is not None:
                    index_cpu_to_gpu_func_fallback_raw: object = getattr(
                        faiss_module, "index_cpu_to_gpu", None
                    )
                    if index_cpu_to_gpu_func_fallback_raw is not None:
                        index_cpu_to_gpu_callable_fallback: Callable[
                            [GpuResourcesProtocol, int, FaissIndexProtocol], object
                        ] = cast(
                            Callable[[GpuResourcesProtocol, int, FaissIndexProtocol], object],
                            index_cpu_to_gpu_func_fallback_raw,
                        )
                        gpu_index_raw_fallback: object = index_cpu_to_gpu_callable_fallback(
                            self._res, 0, cpu_index
                        )
                        self._index = cast(FaissIndexProtocol, gpu_index_raw_fallback)
                    else:
                        self._index = cpu_index
                else:
                    self._index = cpu_index
        else:
            self._index = cpu_index
        try:
            if self.gpu:
                params_raw: object = getattr(faiss_module, "GpuParameterSpace", None)
            else:
                params_raw_cpu: object = getattr(faiss_module, "ParameterSpace", None)
                params_raw = params_raw_cpu
            if params_raw is not None and self._index is not None:
                params_constructor: Callable[[], object] = cast(Callable[[], object], params_raw)
                params = params_constructor()
                set_index_param_func_raw: object = getattr(params, "set_index_parameter", None)
                if set_index_param_func_raw is not None:
                    set_index_param_func: Callable[
                        [object, FaissIndexProtocol, str, object], None
                    ] = cast(
                        Callable[[object, FaissIndexProtocol, str, object], None],
                        set_index_param_func_raw,
                    )
                    set_index_param_func(params, self._index, "nprobe", self.nprobe)
        except (RuntimeError, AttributeError, ValueError) as exc:
            logger.debug("Failed to set nprobe parameter: %s", exc)
            # Continue without parameter setting (FAISS will use defaults)

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
            # np.linalg.norm returns floating[Any] due to numpy typing limitations
            norms_result: NDArray[np.floating[Any]] = np.linalg.norm(
                self._xb, axis=1, keepdims=True
            )
            norms: FloatArray = (norms_result + 1e-12).astype(np.float32)  # type: ignore[misc]
            self._xb /= norms
            return
        faiss_module: FaissModuleProtocol = self._faiss
        if self._index is None:
            message = "FAISS index not initialized; call train() before add()."
            raise RuntimeError(message)
        faiss_module.normalize_L2(vec_array)
        idmap_array: IntArray = np.asarray(keys, dtype=np.int64)
        if isinstance(self._index, FaissIndexIDMap2):
            self._index.add_with_ids(vec_array, idmap_array)
        else:
            index_idmap2_constructor_raw: object = getattr(faiss_module, "IndexIDMap2", None)
            if index_idmap2_constructor_raw is not None:
                index_idmap2_constructor: Callable[[FaissIndexProtocol], FaissIndexProtocol] = cast(
                    Callable[[FaissIndexProtocol], FaissIndexProtocol],
                    index_idmap2_constructor_raw,
                )
                index_with_ids: FaissIndexProtocol = index_idmap2_constructor(self._index)
                index_with_ids.add_with_ids(vec_array, idmap_array)
                self._index = index_with_ids
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
        # np.linalg.norm returns floating[Any] due to numpy typing limitations
        norm_result: NDArray[np.floating[Any]] = np.linalg.norm(q, axis=-1, keepdims=True)
        norm_float32: FloatArray = (norm_result + 1e-12).astype(np.float32)  # type: ignore[misc]
        q /= norm_float32
        if self._faiss is None or self._index is None:
            if self._xb is None or self._idmap is None:
                return []
            # Matrix multiplication returns floating[Any] due to numpy typing limitations
            sims_matrix_result: NDArray[np.floating[Any]] = self._xb @ q.T
            sims_matrix: FloatArray = cast(FloatArray, sims_matrix_result.astype(np.float32))  # type: ignore[misc]
            sims: FloatArray = np.asarray(sims_matrix, dtype=np.float32).squeeze()
            indices: NDArray[np.int64] = np.argsort(-sims)[:k]
            # numpy tolist() returns list[Any] but can be narrowed based on dtype
            # For int64 arrays, tolist() returns list[int] at runtime
            indices_list: list[int] = cast(list[int], indices.tolist())
            # Index sims explicitly - numpy indexing returns Any
            return [(str(self._idmap[i]), float(sims[i])) for i in indices_list]  # type: ignore[misc]
        if self._idmap is None:
            message = "ID map not loaded; cannot resolve FAISS results."
            raise RuntimeError(message)
        distances_result: tuple[NDArray[np.float32], NDArray[np.int64]] = self._index.search(
            q.reshape(1, -1), k
        )
        distances: NDArray[np.float32] = distances_result[0]
        indices_result: NDArray[np.int64] = distances_result[1]
        ids: NDArray[np.int64] = indices_result[0]
        scores: NDArray[np.float32] = distances[0]
        # numpy tolist() returns list[Any] but can be narrowed based on dtype
        # For int64 arrays, tolist() returns list[int] at runtime
        # For float32 arrays, tolist() returns list[float] at runtime
        ids_list: list[int] = cast(list[int], ids.tolist())
        scores_list: list[float] = cast(list[float], scores.tolist())
        return [
            (str(ids_list[i]), scores_list[i]) for i in range(len(ids_list)) if ids_list[i] != -1
        ]

    def save(self, index_uri: str, idmap_uri: str | None = None) -> None:  # noqa: ARG002
        """Describe save.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index_uri : str
            Describe ``index_uri``.
        idmap_uri : str | NoneType, optional
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
        faiss_module: FaissModuleProtocol = self._faiss
        if self.gpu:
            index_gpu_to_cpu_func_raw: object = getattr(faiss_module, "index_gpu_to_cpu", None)
            if index_gpu_to_cpu_func_raw is not None and self._index is not None:
                index_gpu_to_cpu_callable: Callable[[FaissIndexProtocol], FaissIndexProtocol] = (
                    cast(
                        Callable[[FaissIndexProtocol], FaissIndexProtocol],
                        index_gpu_to_cpu_func_raw,
                    )
                )
                target_index: FaissIndexProtocol = index_gpu_to_cpu_callable(self._index)
            else:
                target_index = self._index
        else:
            target_index = self._index
        if target_index is not None:
            write_index_func_raw: object = getattr(faiss_module, "write_index", None)
            if write_index_func_raw is not None:
                write_index_callable: Callable[[FaissIndexProtocol, str], None] = cast(
                    Callable[[FaissIndexProtocol, str], None], write_index_func_raw
                )
                write_index_callable(target_index, index_uri)

    def load(self, index_uri: str, idmap_uri: str | None = None) -> None:  # noqa: ARG002
        """Describe load.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index_uri : str
            Describe ``index_uri``.
        idmap_uri : str | NoneType, optional
            Describe ``idmap_uri``.
            Defaults to ``None``.

        Raises
        ------
        RuntimeError
        Raised when TODO for RuntimeError.
"""
        if self._faiss is None:
            if Path(index_uri + ".npz").exists():
                # np.load returns dict[str, NDArray[Any]] with allow_pickle=True
                # numpy typing limitations require explicit casts
                data_untyped: object = np.load(index_uri + ".npz", allow_pickle=True)
                data: dict[str, NDArray[Any]] = cast(dict[str, NDArray[Any]], data_untyped)  # type: ignore[misc]
                xb_data_raw: NDArray[Any] = data["xb"]  # type: ignore[misc]
                ids_data_raw: NDArray[Any] = data["ids"]  # type: ignore[misc]
                self._xb = cast(FloatArray, xb_data_raw.astype(np.float32))  # type: ignore[misc]
                self._idmap = cast(StrArray, ids_data_raw.astype(np.str_))  # type: ignore[misc]
            return
        faiss_module: FaissModuleProtocol | None = self._faiss
        if faiss_module is None:
            message = "FAISS not available"
            raise RuntimeError(message)
        read_index_func_raw: object = getattr(faiss_module, "read_index", None)
        if read_index_func_raw is None:
            message = "FAISS read_index not available"
            raise RuntimeError(message)
        read_index_callable: Callable[[str], FaissIndexProtocol] = cast(
            Callable[[str], FaissIndexProtocol], read_index_func_raw
        )
        cpu_index: FaissIndexProtocol = read_index_callable(index_uri)
        self._ensure_resources()
        if self.gpu:
            options_raw: object = getattr(faiss_module, "GpuClonerOptions", None)
            if options_raw is not None:
                options_constructor: Callable[[], GpuClonerOptionsProtocol] = cast(
                    Callable[[], GpuClonerOptionsProtocol], options_raw
                )
                options: GpuClonerOptionsProtocol = options_constructor()
                options.use_cuvs = bool(self.cuvs)
                try:
                    index_cpu_to_gpu_func_raw: object = getattr(
                        faiss_module, "index_cpu_to_gpu", None
                    )
                    if index_cpu_to_gpu_func_raw is not None and self._res is not None:
                        index_cpu_to_gpu_callable: Callable[
                            [
                                GpuResourcesProtocol,
                                int,
                                FaissIndexProtocol,
                                GpuClonerOptionsProtocol,
                            ],
                            object,
                        ] = cast(
                            Callable[
                                [
                                    GpuResourcesProtocol,
                                    int,
                                    FaissIndexProtocol,
                                    GpuClonerOptionsProtocol,
                                ],
                                object,
                            ],
                            index_cpu_to_gpu_func_raw,
                        )
                        gpu_index_raw: object = index_cpu_to_gpu_callable(
                            self._res, 0, cpu_index, options
                        )
                        self._index = cast(FaissIndexProtocol, gpu_index_raw)
                    else:
                        self._index = cpu_index
                except (RuntimeError, OSError, ValueError) as exc:
                    logger.debug(
                        "GPU cloning with cuVS failed, falling back to standard GPU: %s", exc
                    )
                    if self._res is not None:
                        index_cpu_to_gpu_func_fallback_raw: object = getattr(
                            faiss_module, "index_cpu_to_gpu", None
                        )
                        if index_cpu_to_gpu_func_fallback_raw is not None:
                            index_cpu_to_gpu_callable_fallback: Callable[
                                [GpuResourcesProtocol, int, FaissIndexProtocol], object
                            ] = cast(
                                Callable[[GpuResourcesProtocol, int, FaissIndexProtocol], object],
                                index_cpu_to_gpu_func_fallback_raw,
                            )
                            gpu_index_raw_fallback: object = index_cpu_to_gpu_callable_fallback(
                                self._res, 0, cpu_index
                            )
                            self._index = cast(FaissIndexProtocol, gpu_index_raw_fallback)
                        else:
                            self._index = cpu_index
                    else:
                        self._index = cpu_index
            else:
                self._index = cpu_index
        else:
            self._index = cpu_index
