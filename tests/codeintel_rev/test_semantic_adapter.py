from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from codeintel_rev.mcp_server.adapters.semantic import semantic_search


class StubDuckDBCatalog:
    """Stub DuckDB catalog for testing."""

    def __init__(self, _db_path: Any, _vectors_dir: Any) -> None:
        """Initialize stub catalog."""
        self._chunk = {
            "uri": "src/module.py",
            "start_line": 0,
            "end_line": 0,
            "preview": "print('hello world')",
        }

    def __enter__(self) -> StubDuckDBCatalog:  # noqa: PYI034
        """Enter context manager.

        Returns
        -------
        StubDuckDBCatalog
            Self instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool:  # pragma: no cover - passthrough
        """Exit context manager.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type.
        exc : BaseException | None
            Exception instance.
        tb : types.TracebackType | None
            Traceback.

        Returns
        -------
        bool
            Always returns False.
        """
        return False

    def get_chunk_by_id(self, chunk_id: int) -> dict[str, Any] | None:
        return self._chunk if chunk_id == 123 else None


class StubVLLMClient:
    """Stub vLLM client for testing."""

    def __init__(self, _config: Any) -> None:
        """Initialize stub vLLM client."""

    def embed_single(self, query: str) -> list[float]:
        assert query
        return [0.1, 0.2]


def _settings(index_path: str, tmp_path: Path) -> Any:
    """Create test settings.

    Parameters
    ----------
    index_path : str
        Path to FAISS index file.
    tmp_path : Path
        Temporary directory for test files.

    Returns
    -------
    Any
        Settings namespace object.
    """
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(
            faiss_index=index_path,
            vectors_dir=str(vectors_dir),
            duckdb_path=str(tmp_path / "catalog.duckdb"),
        ),
        index=SimpleNamespace(vec_dim=2, faiss_nlist=1, use_cuvs=False),
        vllm=SimpleNamespace(),
    )


class _BaseStubFAISSManager:
    """Base stub FAISS manager for testing."""

    def __init__(self, *, should_fail_gpu: bool) -> None:
        """Initialize stub FAISS manager.

        Parameters
        ----------
        should_fail_gpu : bool
            Whether GPU cloning should fail.
        """
        self.should_fail_gpu = should_fail_gpu
        self.gpu_disabled_reason: str | None = None
        self.clone_invocations = 0

    def load_cpu_index(self) -> None:
        """Load CPU index (no-op for testing)."""
        return

    def clone_to_gpu(self) -> bool:
        """Clone to GPU (may fail based on should_fail_gpu).

        Returns
        -------
        bool
            True if GPU cloning succeeds, False otherwise.
        """
        self.clone_invocations += 1
        if self.should_fail_gpu:
            self.gpu_disabled_reason = "FAISS GPU disabled - using CPU: simulated failure"
            return False
        self.gpu_disabled_reason = None
        return True

    def search(
        self, query: np.ndarray, *, k: int, nprobe: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mock search results.

        Parameters
        ----------
        query : np.ndarray
            Query vector.
        k : int
            Number of results to return.
        nprobe : int, optional
            Number of probes. Defaults to 128.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (distances, ids) arrays.
        """
        assert query.shape == (1, 2)
        assert k == 1
        assert nprobe == 128
        return np.array([[0.9]], dtype=np.float32), np.array([[123]], dtype=np.int64)


@pytest.mark.asyncio
async def test_semantic_search_gpu_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    faiss_index_path = tmp_path / "index.faiss"
    faiss_index_path.write_text("dummy")

    class StubFAISSManager(_BaseStubFAISSManager):
        """Stub FAISS manager for GPU success test."""

        def __init__(self, index_path: Path, vec_dim: int, nlist: int, *, use_cuvs: bool) -> None:
            """Initialize stub FAISS manager for GPU success."""
            super().__init__(should_fail_gpu=False)
            assert index_path == faiss_index_path
            assert vec_dim == 2
            assert nlist == 1
            assert use_cuvs is False

    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.FAISSManager",
        StubFAISSManager,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.DuckDBCatalog",
        StubDuckDBCatalog,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.VLLMClient",
        StubVLLMClient,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.load_settings",
        lambda: _settings(str(faiss_index_path), tmp_path),
    )

    result = await semantic_search("hello", limit=1)

    assert "limits" not in result
    findings = result.get("findings")
    assert findings is not None
    assert len(findings) > 0
    location = findings[0].get("location")
    assert location is not None
    assert location.get("uri") == "src/module.py"


@pytest.mark.asyncio
async def test_semantic_search_gpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    faiss_index_path = tmp_path / "index.faiss"
    faiss_index_path.write_text("dummy")

    class StubFAISSManager(_BaseStubFAISSManager):
        """Stub FAISS manager for GPU fallback test."""

        def __init__(self, index_path: Path, vec_dim: int, nlist: int, *, use_cuvs: bool) -> None:
            """Initialize stub FAISS manager for GPU fallback."""
            super().__init__(should_fail_gpu=True)
            assert index_path == faiss_index_path
            assert vec_dim == 2
            assert nlist == 1
            assert use_cuvs is False

    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.FAISSManager",
        StubFAISSManager,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.DuckDBCatalog",
        StubDuckDBCatalog,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.VLLMClient",
        StubVLLMClient,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.load_settings",
        lambda: _settings(str(faiss_index_path), tmp_path),
    )

    result = await semantic_search("hello", limit=1)

    limits = result.get("limits")
    assert limits == ["FAISS GPU disabled - using CPU: simulated failure"]
    findings = result.get("findings")
    assert findings is not None
    assert len(findings) > 0
    location = findings[0].get("location")
    assert location is not None
    assert location.get("uri") == "src/module.py"
