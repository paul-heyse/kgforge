from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from codeintel_rev.mcp_server.adapters.semantic import semantic_search


class StubDuckDBCatalog:
    def __init__(self, _db_path: Any, _vectors_dir: Any) -> None:
        self._chunk = {
            "uri": "src/module.py",
            "start_line": 0,
            "end_line": 0,
            "preview": "print('hello world')",
        }

    def __enter__(self) -> "StubDuckDBCatalog":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - passthrough
        return False

    def get_chunk_by_id(self, chunk_id: int) -> dict[str, Any] | None:
        return self._chunk if chunk_id == 123 else None


class StubVLLMClient:
    def __init__(self, _config: Any) -> None:
        pass

    def embed_single(self, query: str) -> list[float]:
        assert query
        return [0.1, 0.2]


def _settings(index_path: str) -> Any:
    return SimpleNamespace(
        paths=SimpleNamespace(
            faiss_index=index_path,
            vectors_dir="/tmp/vectors",
            duckdb_path="/tmp/catalog.duckdb",
        ),
        index=SimpleNamespace(vec_dim=2, faiss_nlist=1, use_cuvs=False),
        vllm=SimpleNamespace(),
    )


class _BaseStubFAISSManager:
    def __init__(self, *, should_fail_gpu: bool) -> None:
        self.should_fail_gpu = should_fail_gpu
        self.gpu_disabled_reason: str | None = None
        self.clone_invocations = 0

    def load_cpu_index(self) -> None:
        return None

    def clone_to_gpu(self) -> bool:
        self.clone_invocations += 1
        if self.should_fail_gpu:
            self.gpu_disabled_reason = "FAISS GPU disabled - using CPU: simulated failure"
            return False
        self.gpu_disabled_reason = None
        return True

    def search(self, query: np.ndarray, *, k: int, nprobe: int = 128):
        assert query.shape == (1, 2)
        assert k == 1
        assert nprobe == 128
        return np.array([[0.9]], dtype=np.float32), np.array([[123]], dtype=np.int64)


@pytest.mark.asyncio
async def test_semantic_search_gpu_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    faiss_index_path = tmp_path / "index.faiss"
    faiss_index_path.write_text("dummy")

    class StubFAISSManager(_BaseStubFAISSManager):
        def __init__(self, index_path, vec_dim, nlist, *, use_cuvs):
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
        lambda: _settings(str(faiss_index_path)),
    )

    result = await semantic_search("hello", limit=1)

    assert "limits" not in result
    assert result["findings"]
    assert result["findings"][0]["location"]["uri"] == "src/module.py"


@pytest.mark.asyncio
async def test_semantic_search_gpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    faiss_index_path = tmp_path / "index.faiss"
    faiss_index_path.write_text("dummy")

    class StubFAISSManager(_BaseStubFAISSManager):
        def __init__(self, index_path, vec_dim, nlist, *, use_cuvs):
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
        lambda: _settings(str(faiss_index_path)),
    )

    result = await semantic_search("hello", limit=1)

    assert result["limits"] == ["FAISS GPU disabled - using CPU: simulated failure"]
    assert result["findings"]
    assert result["findings"][0]["location"]["uri"] == "src/module.py"
