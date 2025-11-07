from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
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


class StubContext:
    """Stub service context for semantic adapter tests."""

    def __init__(
        self,
        *,
        faiss_manager: _BaseStubFAISSManager,
        limits: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        """Initialize stub context."""
        self.faiss_manager = faiss_manager
        self.vllm_client = StubVLLMClient(SimpleNamespace())
        self.settings = SimpleNamespace()
        self._limits = limits or []
        self._error = error

    def ensure_faiss_ready(self) -> tuple[bool, list[str], str | None]:
        """Return readiness tuple.

        Returns
        -------
        tuple[bool, list[str], str | None]
            Tuple of (ready, limits, error).
        """
        ready = self._error is None
        return ready, list(self._limits), self._error

    @contextmanager
    def open_catalog(self) -> Iterator[StubDuckDBCatalog]:
        """Yield stub catalog.

        Yields
        ------
        StubDuckDBCatalog
            Stub catalog instance.
        """
        yield StubDuckDBCatalog(None, None)


@pytest.mark.asyncio
async def test_semantic_search_gpu_success(monkeypatch: pytest.MonkeyPatch) -> None:
    context = StubContext(
        faiss_manager=_BaseStubFAISSManager(should_fail_gpu=False),
        limits=[],
        error=None,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.get_service_context",
        lambda: context,
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
async def test_semantic_search_gpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    context = StubContext(
        faiss_manager=_BaseStubFAISSManager(should_fail_gpu=True),
        limits=["FAISS GPU disabled - using CPU: simulated failure"],
        error=None,
    )
    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.get_service_context",
        lambda: context,
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
