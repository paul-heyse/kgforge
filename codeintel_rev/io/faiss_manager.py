"""FAISS manager for GPU-accelerated vector search.

Manages IVF-PQ index with cuVS acceleration, CPU persistence, and GPU cloning.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss


class FAISSManager:
    """FAISS index manager with GPU support.

    Parameters
    ----------
    index_path : Path
        Path to CPU index file.
    vec_dim : int
        Vector dimension.
    nlist : int
        Number of IVF centroids.
    use_cuvs : bool
        Enable cuVS acceleration.
    """

    def __init__(
        self,
        index_path: Path,
        vec_dim: int = 2560,
        nlist: int = 8192,
        use_cuvs: bool = True,
    ) -> None:
        self.index_path = index_path
        self.vec_dim = vec_dim
        self.nlist = nlist
        self.use_cuvs = use_cuvs
        self.cpu_index: faiss.Index | None = None
        self.gpu_index: faiss.Index | None = None
        self.gpu_resources: faiss.StandardGpuResources | None = None

    def build_index(self, vectors: np.ndarray) -> None:
        """Build and train IVF-PQ index.

        Parameters
        ----------
        vectors : np.ndarray
            Training vectors of shape (n, vec_dim).

        Raises
        ------
        ImportError
            If FAISS not available.
        """
        try:
            import faiss
        except ImportError as e:
            msg = "FAISS not installed"
            raise ImportError(msg) from e

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        # Create index factory (OPQ + IVF + PQ)
        index_string = f"OPQ64,IVF{self.nlist},PQ64"
        cpu_index = faiss.index_factory(self.vec_dim, index_string, faiss.METRIC_INNER_PRODUCT)

        # Train index
        cpu_index.train(vectors)

        # Wrap in IDMap for ID management
        self.cpu_index = faiss.IndexIDMap2(cpu_index)

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors with IDs to the index.

        Adds a batch of vectors to the FAISS index with their associated IDs.
        The vectors are normalized for cosine similarity (L2 normalization) before
        being added. IDs are used for retrieval - they should match the chunk IDs
        stored in DuckDB.

        This method requires that build_index() has been called first to create
        and train the index structure.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors to add, shape (n, vec_dim) where n is the number of vectors
            and vec_dim matches the index dimension. Dtype should be float32.
        ids : np.ndarray
            Unique IDs for each vector, shape (n,). IDs are stored as int64 in
            FAISS. These should correspond to chunk IDs from the indexing pipeline.

        Raises
        ------
        RuntimeError
            If the index has not been built yet. Call build_index() first.
        ImportError
            If the FAISS library is not installed or cannot be imported. Install
            with `uv add faiss` or `uv add faiss-cpu` for CPU-only version.
        """
        if self.cpu_index is None:
            msg = "Index not built"
            raise RuntimeError(msg)

        try:
            import faiss
        except ImportError as e:
            msg = "FAISS not installed"
            raise ImportError(msg) from e

        # Normalize vectors
        vectors_norm = vectors.copy()
        faiss.normalize_L2(vectors_norm)

        # Add with IDs
        self.cpu_index.add_with_ids(vectors_norm, ids.astype(np.int64))

    def save_cpu_index(self) -> None:
        """Save CPU index to disk for persistence.

        Writes the current CPU index to the file specified by index_path. The
        index can be loaded later with load_cpu_index() to avoid rebuilding.
        The parent directory is created if it doesn't exist.

        The saved index includes all vectors and IDs that have been added. This
        is the CPU version - GPU indexes are cloned on-demand and not persisted.

        Raises
        ------
        RuntimeError
            If the index has not been built yet. Call build_index() first.
        ImportError
            If the FAISS library is not installed or cannot be imported. Install
            with `uv add faiss` or `uv add faiss-cpu` for CPU-only version.
        """
        if self.cpu_index is None:
            msg = "Index not built"
            raise RuntimeError(msg)

        try:
            import faiss
        except ImportError as e:
            msg = "FAISS not installed"
            raise ImportError(msg) from e

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.cpu_index, str(self.index_path))

    def load_cpu_index(self) -> None:
        """Load CPU index from disk.

        Reads a previously saved FAISS index from index_path and loads it into
        memory. This allows reusing an index without rebuilding it, which is much
        faster for large indexes.

        After loading, you can call clone_to_gpu() to create a GPU version for
        faster search, or use search() directly with the CPU index.

        Raises
        ------
        FileNotFoundError
            If the index file does not exist at index_path. Ensure the index was
            saved with save_cpu_index() or that the path is correct.
        ImportError
            If the FAISS library is not installed or cannot be imported. Install
            with `uv add faiss` or `uv add faiss-cpu` for CPU-only version.
        """
        if not self.index_path.exists():
            msg = f"Index not found: {self.index_path}"
            raise FileNotFoundError(msg)

        try:
            import faiss
        except ImportError as e:
            msg = "FAISS not installed"
            raise ImportError(msg) from e

        self.cpu_index = faiss.read_index(str(self.index_path))

    def clone_to_gpu(self, device: int = 0) -> None:
        """Clone CPU index to GPU for accelerated search.

        Creates a GPU-resident copy of the CPU index for faster search operations.
        The GPU index uses the same structure (IVF-PQ) but runs on GPU hardware
        for 10-100x speedup on large indexes.

        If cuVS acceleration is enabled (use_cuvs=True), the function attempts to
        use optimized cuVS kernels. If cuVS is unavailable, it falls back to
        standard FAISS GPU operations.

        The GPU index is kept in memory alongside the CPU index. Both can be
        used for search, but GPU is preferred when available.

        Parameters
        ----------
        device : int, optional
            CUDA device ID to use (default: 0). Use device 0 for single-GPU systems.
            For multi-GPU, specify the device ID (0, 1, 2, etc.).

        Raises
        ------
        RuntimeError
            If the CPU index has not been loaded yet. Call load_cpu_index() or
            build_index() first. Also raised if GPU operations fail (e.g., CUDA
            not available, out of GPU memory).
        ImportError
            If the FAISS library is not installed or cannot be imported. Install
            with `uv add faiss` (GPU version) or ensure CUDA is properly configured.
        """
        if self.cpu_index is None:
            msg = "CPU index not loaded"
            raise RuntimeError(msg)

        try:
            import faiss
        except ImportError as e:
            msg = "FAISS not installed"
            raise ImportError(msg) from e

        # Initialize GPU resources
        self.gpu_resources = faiss.StandardGpuResources()

        # Configure cloner options
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False

        # Try with cuVS if enabled
        if self.use_cuvs:
            try:
                from pylibcuvs import load_library

                load_library()
                co.use_cuvs = True
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources, device, self.cpu_index, co
                )
                return
            except (ImportError, RuntimeError, AttributeError):
                # Fall back to standard GPU
                pass

        # Standard GPU clone
        co.use_cuvs = False
        self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, device, self.cpu_index, co)

    def search(
        self, query: np.ndarray, k: int = 50, nprobe: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors using cosine similarity.

        Performs approximate nearest neighbor search using the FAISS index.
        The query vector(s) are normalized for cosine similarity, then searched
        against the indexed vectors.

        The function automatically uses the GPU index if available (faster),
        otherwise falls back to CPU. The nprobe parameter controls the trade-off
        between search speed and recall - higher values search more cells and
        improve recall but slow down search.

        Parameters
        ----------
        query : np.ndarray
            Query vector(s) of shape (n_queries, vec_dim) or (vec_dim,) for
            single query. Dtype should be float32. Vectors are automatically
            normalized for cosine similarity.
        k : int, optional
            Number of nearest neighbors to return per query (default: 50).
            Higher k improves recall but increases computation and memory usage.
        nprobe : int, optional
            Number of IVF cells to probe during search (default: 128). Higher
            values improve recall but slow down search. Should match or be less
            than the nlist parameter used during index construction.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (distances, ids) arrays:
            - distances: shape (n_queries, k), cosine similarity scores (higher
              is more similar, range typically 0-1 after normalization)
            - ids: shape (n_queries, k), chunk IDs of the nearest neighbors.
              IDs correspond to the ids passed to add_vectors().

        Raises
        ------
        RuntimeError
            If no index is available (neither CPU nor GPU index loaded). Call
            load_cpu_index() or build_index() first. Also raised if search fails
            (e.g., dimension mismatch, invalid nprobe value).
        ImportError
            If the FAISS library is not installed or cannot be imported. Install
            with `uv add faiss` or `uv add faiss-cpu` for CPU-only version.
        """
        if self.gpu_index is None and self.cpu_index is None:
            msg = "No index available"
            raise RuntimeError(msg)

        try:
            import faiss
        except ImportError as e:
            msg = "FAISS not installed"
            raise ImportError(msg) from e

        # Use GPU if available, otherwise CPU
        index = self.gpu_index if self.gpu_index is not None else self.cpu_index

        # Set nprobe
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe

        # Reshape query if needed
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        query_norm = query.copy().astype(np.float32)
        faiss.normalize_L2(query_norm)

        # Search
        distances, ids = index.search(query_norm, k)

        return distances, ids


__all__ = ["FAISSManager"]
