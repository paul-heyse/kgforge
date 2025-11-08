from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from codeintel_rev.app.config_context import ApplicationContext

from codeintel_rev.mcp_server.adapters.semantic import semantic_search


@pytest.fixture(autouse=True)
def _stub_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal metrics stub for semantic adapter tests."""

    class _Histogram:
        def labels(self, **_: object) -> _Histogram:
            return self

        def observe(self, _: object) -> None:
            return None

    class _Counter:
        def labels(self, **_: object) -> _Counter:
            return self

        def inc(self, _: object | None = None) -> None:
            return None

    class _Metrics:
        operation_duration_seconds = _Histogram()
        runs_total = _Counter()

    monkeypatch.setattr(
        "codeintel_rev.mcp_server.adapters.semantic.METRICS",
        _Metrics(),
    )


class StubDuckDBCatalog:
    """Stub DuckDB catalog for testing."""

    def __init__(self, _db_path: Any, _vectors_dir: Any) -> None:
        """Initialize stub catalog."""
        self._chunk = {
            "id": 123,
            "uri": "src/module.py",
            "start_line": 0,
            "end_line": 0,
            "preview": "print('hello world')",
        }

    def __enter__(self) -> Self:
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

    def query_by_ids(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        if 123 in chunk_ids:
            return [dict(self._chunk)]
        return []


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
        self.last_k: int | None = None

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
        assert k >= 1
        self.last_k = k
        assert nprobe == 128
        return np.array([[0.9]], dtype=np.float32), np.array([[123]], dtype=np.int64)


class StubContext:
    """Stub ApplicationContext for semantic adapter tests."""

    def __init__(
        self,
        *,
        faiss_manager: _BaseStubFAISSManager,
        limits: list[str] | None = None,
        error: str | None = None,
        max_results: int = 5,
    ) -> None:
        """Initialize stub context."""
        self.faiss_manager = faiss_manager
        self.vllm_client = StubVLLMClient(SimpleNamespace())
        self.settings = SimpleNamespace(
            limits=SimpleNamespace(max_results=max_results),
            vllm=SimpleNamespace(base_url="http://localhost"),
        )
        # Use tempfile for secure temporary paths in tests
        import tempfile

        temp_dir = Path(tempfile.gettempdir())
        self.paths = SimpleNamespace(
            faiss_index=temp_dir / "index.faiss",
            duckdb_path=temp_dir / "catalog.duckdb",
            vectors_dir=temp_dir / "vectors",
        )
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
async def test_semantic_search_gpu_success() -> None:
    context = StubContext(
        faiss_manager=_BaseStubFAISSManager(should_fail_gpu=False),
        limits=[],
        error=None,
    )

    # Cast StubContext to ApplicationContext for type checking
    # StubContext implements the necessary interface for testing
    result = await semantic_search(cast("ApplicationContext", context), "hello", limit=1)

    assert "limits" not in result
    findings = result.get("findings")
    assert findings is not None
    assert len(findings) > 0
    location = findings[0].get("location")
    assert location is not None
    assert location.get("uri") == "src/module.py"


@pytest.mark.asyncio
async def test_semantic_search_gpu_fallback() -> None:
    context = StubContext(
        faiss_manager=_BaseStubFAISSManager(should_fail_gpu=True),
        limits=["FAISS GPU disabled - using CPU: simulated failure"],
        error=None,
    )

    result = await semantic_search(cast("ApplicationContext", context), "hello", limit=1)

    limits = result.get("limits")
    assert limits == ["FAISS GPU disabled - using CPU: simulated failure"]
    findings = result.get("findings")
    assert findings is not None
    assert len(findings) > 0
    location = findings[0].get("location")
    assert location is not None
    assert location.get("uri") == "src/module.py"


@pytest.mark.asyncio
async def test_semantic_search_limit_truncates_to_max_results() -> None:
    faiss_manager = _BaseStubFAISSManager(should_fail_gpu=False)
    context = StubContext(
        faiss_manager=faiss_manager,
        limits=[],
        error=None,
        max_results=3,
    )

    result = await semantic_search(cast("ApplicationContext", context), "hello", limit=10)

    assert faiss_manager.last_k == 3
    limits = result.get("limits")
    assert limits is not None
    assert any("exceeds max_results" in message for message in limits)
    method = result.get("method")
    assert method is not None
    coverage = method.get("coverage")
    assert coverage is not None
    assert "/3 results" in coverage
    assert "requested 10" in coverage


@pytest.mark.asyncio
async def test_semantic_search_limit_enforces_minimum() -> None:
    faiss_manager = _BaseStubFAISSManager(should_fail_gpu=False)
    context = StubContext(
        faiss_manager=faiss_manager,
        limits=[],
        error=None,
        max_results=5,
    )

    result = await semantic_search(cast("ApplicationContext", context), "hello", limit=0)

    assert faiss_manager.last_k == 1
    limits = result.get("limits")
    assert limits is not None
    assert any("not positive" in message for message in limits)
    method = result.get("method")
    assert method is not None
    coverage = method.get("coverage")
    assert coverage is not None
    assert "/1 results" in coverage
    assert "requested 0" in coverage
