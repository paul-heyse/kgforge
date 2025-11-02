"""FAISS adapter with typed GPU fallbacks and DuckDB integration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeGuard, cast

import duckdb
import numpy as np
from numpy.typing import NDArray

from kgfoundry_common.errors import IndexBuildError, VectorSearchError
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.numpy_typing import (
    FloatMatrix,
    FloatVector,
    IntVector,
    normalize_l2,
    topk_indices,
)
from search_api.faiss_gpu import (
    GpuContext,
    clone_index_to_gpu,
    configure_search_parameters,
    detect_gpu_context,
)
from search_api.types import FaissIndexProtocol, FaissModuleProtocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

    type FloatArray = npt.NDArray[np.float32]
    type IntArray = npt.NDArray[np.int64]
    type StrArray = npt.NDArray[np.str_]
    type VecArray = npt.NDArray[np.float32]
else:  # pragma: no cover - runtime fallback
    FloatArray = np.ndarray
    IntArray = np.ndarray
    StrArray = np.ndarray
    VecArray = np.ndarray

__all__ = ["DenseVecs", "FaissAdapter", "FloatArray", "IntArray", "StrArray", "VecArray"]

__navmap__: Final[NavMap] = {
    "title": "search_api.faiss_adapter",
    "synopsis": "FAISS adapter with typed GPU fallbacks and DuckDB persistence",
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

logger = logging.getLogger(__name__)


def _is_faiss_index(candidate: object) -> TypeGuard[FaissIndexProtocol]:
    """Return True when ``candidate`` exposes the FAISS index protocol surface."""
    if candidate is None:
        return False

    required_methods = ("search", "add", "train", "add_with_ids")
    for method in required_methods:
        attribute: object | None = getattr(candidate, method, None)
        if attribute is None or not callable(attribute):
            return False
    return True


MIN_FACTORY_DIMENSION: Final[int] = 64

try:  # pragma: no cover - optional dependency
    from libcuvs import load_library as _load_cuvs

    _load_cuvs()
except (ImportError, RuntimeError, OSError):  # pragma: no cover - optional dependency
    logger.debug("cuVS library not available; continuing with FAISS CPU helpers")

try:  # pragma: no cover - optional dependency
    import faiss as _faiss_module
except (ImportError, ModuleNotFoundError, OSError) as exc:  # pragma: no cover - optional dependency
    logger.debug("FAISS import failed: %s", exc)
    faiss: FaissModuleProtocol | None = None
    HAVE_FAISS = False
else:
    faiss = cast(FaissModuleProtocol, _faiss_module)
    HAVE_FAISS = True


@dataclass(slots=True)
class DenseVecs:
    """Dense vector matrix and ID mapping used to seed FAISS indexes."""

    ids: list[str]
    matrix: FloatMatrix


class FaissAdapter:
    """Build FAISS indexes with optional GPU acceleration and CPU fallback."""

    def __init__(  # noqa: PLR0913 - standard __init__ with configuration options
        self,
        db_path: str,
        *,
        factory: str = "OPQ64,IVF8192,PQ64",
        metric: str = "ip",
        nprobe: int = 64,
        use_gpu: bool = True,
        use_cuvs: bool = True,
        gpu_devices: Sequence[int] | None = None,
    ) -> None:
        self.db_path = db_path
        self.factory = factory
        self.metric = metric
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.use_cuvs = use_cuvs
        self._gpu_devices = tuple(int(device) for device in (gpu_devices or (0,)))

        self.index: FaissIndexProtocol | None = None
        self.idmap: list[str] | None = None
        self.vecs: DenseVecs | None = None

        self._cpu_matrix: FloatMatrix | None = None
        self._gpu_context: GpuContext | None = None

    def build(self) -> None:
        """Build or rebuild the FAISS index from persisted vectors.

        Raises
        ------
        IndexBuildError
            If index construction fails.
        """
        vectors = self._load_dense_vectors()
        self.vecs = vectors
        self.idmap = vectors.ids
        self._cpu_matrix = vectors.matrix

        module = faiss
        if not HAVE_FAISS or module is None:
            logger.debug("FAISS unavailable; CPU search fallback will be used")
            self.index = None
            return
        faiss_module: FaissModuleProtocol = module

        try:
            dimension = vectors.matrix.shape[1]
            metric_type = self._resolve_metric(faiss_module)
            factory = self.factory if dimension >= MIN_FACTORY_DIMENSION else "Flat"

            cpu_index = faiss_module.index_factory(dimension, factory, metric_type)

            faiss_module.normalize_L2(vectors.matrix)
            cpu_index.train(vectors.matrix)

            id_array = cast(IntVector, np.arange(len(vectors.ids), dtype=np.int64))
            cpu_index.add_with_ids(vectors.matrix, id_array)

            gpu_context = None
            index: FaissIndexProtocol = cpu_index
            if self.use_gpu:
                gpu_context = detect_gpu_context(
                    faiss_module,
                    use_cuvs=self.use_cuvs,
                    device_ids=self._gpu_devices,
                )
                if gpu_context is not None:
                    index = clone_index_to_gpu(cpu_index, gpu_context)

            configure_search_parameters(
                faiss_module,
                index,
                nprobe=self.nprobe,
                gpu_enabled=gpu_context is not None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"Failed to build FAISS index: {exc}"
            raise IndexBuildError(msg) from exc

        if not _is_faiss_index(index):
            msg = "FAISS index failed protocol validation"
            raise IndexBuildError(msg)

        self.index = index
        self._gpu_context = gpu_context

    def load_or_build(self, cpu_index_path: str | None = None) -> None:
        """Load an existing CPU index or fall back to rebuilding from vectors.

        Raises
        ------
        VectorSearchError
            If index loading or vector loading fails.
        """
        module = faiss
        if module is None or not HAVE_FAISS:
            self.build()
            return
        faiss_module: FaissModuleProtocol = module

        if cpu_index_path:
            index_path = Path(cpu_index_path)
            if index_path.exists():
                try:
                    cpu_index = faiss_module.read_index(str(index_path))
                    vectors = self._load_dense_vectors()
                    self.vecs = vectors
                    self.idmap = vectors.ids
                    self._cpu_matrix = vectors.matrix

                    gpu_context = None
                    if self.use_gpu:
                        gpu_context = detect_gpu_context(
                            faiss_module,
                            use_cuvs=self.use_cuvs,
                            device_ids=self._gpu_devices,
                        )
                        if gpu_context is not None:
                            cpu_index = clone_index_to_gpu(cpu_index, gpu_context)
                except (
                    RuntimeError,
                    OSError,
                    ValueError,
                ) as exc:  # pragma: no cover - defensive fallback
                    logger.warning(
                        "Failed to load FAISS index from %s: %s", index_path, exc, exc_info=True
                    )
                else:
                    configure_search_parameters(
                        faiss_module,
                        cpu_index,
                        nprobe=self.nprobe,
                        gpu_enabled=gpu_context is not None,
                    )

                    if not _is_faiss_index(cpu_index):
                        logger.warning(
                            "Loaded FAISS index failed protocol validation; rebuilding",
                        )
                    else:
                        self.index = cpu_index
                        self._gpu_context = gpu_context
                        return

        self.build()

    def search(
        self, query: Sequence[float] | NDArray[np.float32], k: int
    ) -> list[tuple[str, float]]:
        """Return the top ``k`` vector matches for ``query``.

        Parameters
        ----------
        query : Sequence[float] | NDArray[np.float32]
            Query vector.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (doc_id, score) tuples sorted by score descending.

        Raises
        ------
        ValueError
            If k is not positive.
        """
        if k <= 0:
            msg = "k must be positive"
            raise ValueError(msg)

        query_array = cast(FloatMatrix, np.asarray(query, dtype=np.float32).reshape(1, -1))
        normalized_query = normalize_l2(query_array, axis=1)

        module = faiss
        if module is not None and HAVE_FAISS:
            index_candidate = self._require_index()
            if self.idmap is None:
                msg = "ID map not initialized"
                raise RuntimeError(msg)

            distances_array, indices_array = index_candidate.search(normalized_query, k)
            distance_row = cast(FloatVector, distances_array[0])
            index_row = cast(IntVector, indices_array[0])
            index_list = cast(list[int], index_row.tolist())
            score_list = cast(list[float], distance_row.tolist())
            results: list[tuple[str, float]] = []
            for idx, score in zip(index_list, score_list, strict=False):
                if idx < 0 or idx >= len(self.idmap):
                    continue
                results.append((self.idmap[idx], float(score)))
            return results

        normalized_vector = cast(FloatVector, normalized_query[0])
        return self._cpu_search(normalized_vector, k)

    def save(self, index_uri: str, idmap_uri: str | None = None) -> None:
        """Persist the index (when available) and ID mapping to disk.

        Parameters
        ----------
        index_uri : str
            Path where the FAISS index will be saved.
        idmap_uri : str | None, optional
            Path where the ID mapping will be saved. If None, defaults to ``{index_uri}.ids.npy``.

        Raises
        ------
        RuntimeError
            If no vectors have been loaded.
        """
        if self.vecs is None:
            msg = "No vectors loaded; call build() before save()."
            raise RuntimeError(msg)

        idmap_path = Path(idmap_uri or f"{index_uri}.ids.npy")
        idmap_path.parent.mkdir(parents=True, exist_ok=True)
        idmap_array: StrArray = np.asarray(self.vecs.ids, dtype=np.str_)
        np.save(idmap_path, idmap_array)

        module = faiss
        if module is None or not HAVE_FAISS:
            msg = "FAISS module not available; cannot save index"
            raise RuntimeError(msg)

        index_candidate = self._require_index()
        module.write_index(index_candidate, index_uri)

    # Internal helpers -------------------------------------------------------

    def _require_index(self) -> FaissIndexProtocol:
        """Return the initialized FAISS index or raise."""
        index_candidate = self.index
        if index_candidate is None:
            msg = "FAISS index not initialized"
            raise RuntimeError(msg)

        if not _is_faiss_index(index_candidate):
            msg = "FAISS index failed protocol validation"
            raise RuntimeError(msg)

        return index_candidate

    def _cpu_search(self, query: FloatVector, k: int) -> list[tuple[str, float]]:
        """Search using CPU fallback (inner product)."""
        cpu_matrix = self._cpu_matrix
        idmap = self.idmap
        if cpu_matrix is None or idmap is None:  # pragma: no cover - defensive fallback
            return []

        matrix: FloatMatrix = cpu_matrix
        vector: FloatVector = query
        scores_buffer: FloatVector = np.empty(matrix.shape[0], dtype=np.float32)
        np.dot(matrix, vector, out=scores_buffer)
        scores: FloatVector = scores_buffer
        idmap_list: list[str] = idmap
        score_list = cast(list[float], scores.tolist())
        limit = min(k, scores.size)
        if limit == 0:
            return []

        indices = topk_indices(scores, limit)
        index_list = cast(list[int], indices.tolist())
        return [
            (idmap_list[idx], float(score_list[idx])) for idx in index_list if idx < len(idmap_list)
        ]

    def _resolve_metric(self, module: FaissModuleProtocol) -> int:
        """Resolve metric string to FAISS metric constant.

        Raises
        ------
        ValueError
            If metric is not recognized.
        """
        metric = self.metric.lower()
        if metric == "ip":
            return module.METRIC_INNER_PRODUCT
        if metric == "l2":
            return module.METRIC_L2
        msg = f"Unsupported FAISS metric: {self.metric}"
        raise ValueError(msg)

    def _load_dense_vectors(self) -> DenseVecs:
        """Load dense vectors from DuckDB or Parquet.

        Raises
        ------
        VectorSearchError
            If vectors cannot be loaded.
        """
        candidate = Path(self.db_path)
        if candidate.is_dir() or candidate.suffix == ".parquet":
            return self._load_from_parquet(candidate)

        if not candidate.exists():
            msg = f"DuckDB registry not found: {candidate}"
            raise VectorSearchError(msg)

        try:
            con = duckdb.connect(str(candidate))
        except duckdb.Error:
            return self._load_from_parquet(candidate)

        try:
            record = con.execute(
                "SELECT parquet_root FROM dense_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        except duckdb.Error as exc:  # pragma: no cover - defensive fallback
            msg = f"Failed to query dense_runs: {exc}"
            raise VectorSearchError(msg) from exc
        finally:
            con.close()

        if not record or not isinstance(record[0], str):
            msg = "dense_runs table is empty or malformed"
            raise VectorSearchError(msg)

        return self._load_from_parquet(Path(record[0]))

    @staticmethod
    def _load_from_parquet(source: Path) -> DenseVecs:
        """Load dense vectors from Parquet file.

        Raises
        ------
        VectorSearchError
            If Parquet file cannot be read or is empty.
        """
        resolved = source.resolve()
        if not resolved.exists():
            msg = f"Parquet source not found: {resolved}"
            raise VectorSearchError(msg)

        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                "SELECT chunk_id, vector FROM read_parquet(?, union_by_name=true)",
                [str(resolved)],
            ).fetchall()
        except duckdb.Error as exc:
            msg = f"Failed to load vectors from {resolved}: {exc}"
            raise VectorSearchError(msg) from exc
        finally:
            con.close()

        if not rows:
            msg = f"No vectors discovered in {resolved}"
            raise VectorSearchError(msg)

        ids = [str(row[0]) for row in rows]
        vector_rows = [np.asarray(row[1], dtype=np.float32) for row in rows]
        matrix = np.vstack(vector_rows)
        normalized = normalize_l2(matrix, axis=1)
        return DenseVecs(ids=ids, matrix=normalized)
