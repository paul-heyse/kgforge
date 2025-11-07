"""Singleton service context for MCP adapters."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock

from codeintel_rev.config.settings import Settings, load_settings
from codeintel_rev.io.duckdb_catalog import DuckDBCatalog
from codeintel_rev.io.faiss_manager import FAISSManager
from codeintel_rev.io.vllm_client import VLLMClient


def resolve_configured_path(settings: Settings, path_value: str) -> Path:
    """Return an absolute path for ``path_value`` using ``settings.paths.repo_root``.

    Parameters
    ----------
    settings : Settings
        Loaded application settings containing the repository root.
    path_value : str
        Configured path value that may be relative to the repository root.

    Returns
    -------
    Path
        Absolute path pointing to the configured location.
    """
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


@dataclass(slots=True)
class ServiceContext:
    """Container for long-lived service clients."""

    settings: Settings
    vllm_client: VLLMClient
    faiss_manager: FAISSManager
    _faiss_lock: Lock = field(default_factory=Lock, init=False)
    _faiss_loaded: bool = field(default=False, init=False)
    _faiss_gpu_attempted: bool = field(default=False, init=False)

    def ensure_faiss_ready(self) -> tuple[bool, list[str], str | None]:
        """Load FAISS index (once) and attempt GPU clone.

        Returns
        -------
        tuple[bool, list[str], str | None]
            Tuple of (ready, limits, error). ``ready`` is True if FAISS is available,
            ``limits`` contains warnings about GPU availability, and ``error`` is
            None on success or an error message on failure.
        """
        limits: list[str] = []
        index_path = resolve_configured_path(self.settings, self.settings.paths.faiss_index)
        if not index_path.exists():
            return False, limits, f"FAISS index not found at {index_path}"

        with self._faiss_lock:
            if not self._faiss_loaded:
                try:
                    self.faiss_manager.load_cpu_index()
                except (FileNotFoundError, RuntimeError) as exc:
                    return False, limits, f"FAISS index load failed: {exc}"
                self._faiss_loaded = True

            if self.faiss_manager.gpu_index is None and not self._faiss_gpu_attempted:
                self._faiss_gpu_attempted = True
                try:
                    gpu_enabled = self.faiss_manager.clone_to_gpu()
                except RuntimeError as exc:
                    limits.append(str(exc))
                    gpu_enabled = False
                if not gpu_enabled and self.faiss_manager.gpu_disabled_reason:
                    limits.append(self.faiss_manager.gpu_disabled_reason)
            elif (
                self.faiss_manager.gpu_disabled_reason
                and limits.count(self.faiss_manager.gpu_disabled_reason) == 0
            ):
                limits.append(self.faiss_manager.gpu_disabled_reason)

        return True, limits, None

    def reset_faiss(self) -> None:
        """Reset FAISS caches (used for tests or re-index).

        Returns
        -------
        None
        """
        with self._faiss_lock:
            self.faiss_manager.cpu_index = None
            self.faiss_manager.gpu_index = None
            self.faiss_manager.gpu_resources = None
            self.faiss_manager.gpu_disabled_reason = None
            self._faiss_loaded = False
            self._faiss_gpu_attempted = False

    @contextmanager
    def open_catalog(self) -> Iterator[DuckDBCatalog]:
        """Yield a DuckDBCatalog context manager.

        Yields
        ------
        DuckDBCatalog
            Catalog instance for querying chunk metadata.
        """
        catalog = DuckDBCatalog(
            resolve_configured_path(self.settings, self.settings.paths.duckdb_path),
            resolve_configured_path(self.settings, self.settings.paths.vectors_dir),
        )
        try:
            catalog.open()
            yield catalog
        finally:
            catalog.close()


@lru_cache(maxsize=1)
def get_service_context() -> ServiceContext:
    """Return the cached service context.

    Returns
    -------
    ServiceContext
        Cached service context instance.
    """
    settings = load_settings()
    faiss_manager = FAISSManager(
        index_path=resolve_configured_path(settings, settings.paths.faiss_index),
        vec_dim=settings.index.vec_dim,
        nlist=settings.index.faiss_nlist,
        use_cuvs=settings.index.use_cuvs,
    )
    vllm_client = VLLMClient(settings.vllm)
    return ServiceContext(settings=settings, vllm_client=vllm_client, faiss_manager=faiss_manager)


def reset_service_context() -> None:
    """Clear cached context (primarily for tests)."""
    get_service_context.cache_clear()
