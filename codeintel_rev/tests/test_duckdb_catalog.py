from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

from codeintel_rev.io.duckdb_catalog import DuckDBCatalog


def _write_chunk_parquet(parquet_path: Path) -> None:
    conn = duckdb.connect()
    try:
        conn.execute(
            """
            CREATE TABLE chunks AS
            SELECT * FROM (VALUES
                (1, 'src/example.py', 'Example preview', 10, 20)
            ) AS t(id, uri, preview, start_line, end_line)
            """
        )
        conn.execute(
            "COPY chunks TO ? (FORMAT PARQUET)",
            [str(parquet_path)],
        )
    finally:
        conn.close()


def test_get_chunk_by_id_returns_single_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    parquet_path = vectors_dir / "chunks.parquet"
    _write_chunk_parquet(parquet_path)

    def _ensure_views_for_test(self: DuckDBCatalog) -> None:
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        parquet_pattern = str(self.vectors_dir / "**/*.parquet")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW chunks AS SELECT * FROM read_parquet('{parquet_pattern}')",
        )

    monkeypatch.setattr(DuckDBCatalog, "_ensure_views", _ensure_views_for_test, raising=False)

    catalog_path = tmp_path / "catalog.duckdb"
    with DuckDBCatalog(catalog_path, vectors_dir) as catalog:
        chunk = catalog.get_chunk_by_id(1)
        assert chunk is not None
        assert chunk["uri"] == "src/example.py"
        assert chunk["preview"] == "Example preview"
        assert catalog.get_chunk_by_id(9999) is None


@pytest.mark.asyncio
async def test_semantic_search_handles_chunk_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace())

    from codeintel_rev.mcp_server.adapters import semantic

    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    faiss_index = tmp_path / "index.faiss"
    faiss_index.write_bytes(b"faiss")
    duckdb_path = tmp_path / "catalog.duckdb"
    duckdb_path.write_bytes(b"")

    class DummySettings:
        def __init__(self) -> None:
            self.paths = SimpleNamespace(
                faiss_index=str(faiss_index),
                duckdb_path=str(duckdb_path),
                vectors_dir=str(vectors_dir),
            )
            self.index = SimpleNamespace(vec_dim=3, faiss_nlist=1, use_cuvs=False)
            self.vllm = SimpleNamespace()

    class FakeVLLMClient:
        def __init__(self, _config: object) -> None:
            self.calls: list[str] = []

        def embed_single(self, query: str) -> list[float]:
            self.calls.append(query)
            return [0.1, 0.2, 0.3]

    class FakeFAISSManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def load_cpu_index(self) -> None:  # pragma: no cover - trivial
            return None

        def clone_to_gpu(self) -> None:  # pragma: no cover - trivial
            return None

        def search(self, _query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            return np.array([[0.123]]), np.array([[42]])

    class FakeDuckDBCatalog:
        calls: list[int] = []

        def __init__(self, db_path: Path, vectors: Path) -> None:
            self.db_path = db_path
            self.vectors = vectors

        def __enter__(self) -> FakeDuckDBCatalog:
            FakeDuckDBCatalog.calls.clear()
            return self

        def __exit__(self, *_exc: object) -> None:  # pragma: no cover - trivial
            return None

        def get_chunk_by_id(self, chunk_id: int) -> dict | None:
            FakeDuckDBCatalog.calls.append(chunk_id)
            if chunk_id == 42:
                return {
                    "uri": "src/demo.py",
                    "start_line": 1,
                    "end_line": 5,
                    "preview": "semantic chunk",
                }
            return None

    monkeypatch.setattr(semantic, "load_settings", lambda: DummySettings())
    monkeypatch.setattr(semantic, "VLLMClient", FakeVLLMClient)
    monkeypatch.setattr(semantic, "FAISSManager", FakeFAISSManager)
    monkeypatch.setattr(semantic, "DuckDBCatalog", FakeDuckDBCatalog)

    result = await semantic.semantic_search("test query", limit=1)

    assert result["findings"]
    finding = result["findings"][0]
    assert finding["snippet"] == "semantic chunk"
    assert finding["location"]["uri"] == "src/demo.py"
    assert FakeDuckDBCatalog.calls == [42]
