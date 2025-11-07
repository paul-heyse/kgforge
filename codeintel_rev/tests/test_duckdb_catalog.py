"""Tests covering DuckDB catalog helpers and semantic search hydration."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Self

import duckdb
import numpy as np
import pytest

from codeintel_rev.io.duckdb_catalog import DuckDBCatalog
from codeintel_rev.mcp_server.adapters import semantic

MATCHING_CHUNK_ID = 42
MATCHING_PREVIEW = "semantic chunk"


def _expect(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


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


class DummySettings:
    """Minimal settings wrapper supplying adapter configuration."""

    def __init__(self, faiss_index: Path, duckdb_path: Path, vectors_dir: Path) -> None:
        self.paths = SimpleNamespace(
            faiss_index=str(faiss_index),
            duckdb_path=str(duckdb_path),
            vectors_dir=str(vectors_dir),
        )
        self.index = SimpleNamespace(vec_dim=3, faiss_nlist=1, use_cuvs=False)
        self.vllm = SimpleNamespace()


class FakeVLLMClient:
    """Record embedding requests made during semantic search."""

    def __init__(self, _config: object) -> None:
        self.calls: list[str] = []

    def embed_single(self, query: str) -> list[float]:
        """Return a deterministic embedding vector for *query*.

        Returns
        -------
        list[float]
            Mock embedding vector ``[0.1, 0.2, 0.3]``.
        """
        self.calls.append(query)
        return [0.1, 0.2, 0.3]


class FakeFAISSManager:
    """Provide deterministic FAISS results for tests."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.gpu_disabled_reason: str | None = None

    @staticmethod
    def load_cpu_index() -> None:  # pragma: no cover - trivial shim
        """Pretend to load the FAISS CPU index."""
        return

    @staticmethod
    def clone_to_gpu() -> bool:  # pragma: no cover - trivial shim
        """Report that GPU cloning succeeded.

        Returns
        -------
        bool
            Always ``True`` to indicate success.
        """
        return True

    @staticmethod
    def search(_query_vec: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``k`` mock search results.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair of distances and chunk identifiers.
        """
        _ = k
        return (
            np.array([[0.123]], dtype=np.float32),
            np.array([[MATCHING_CHUNK_ID]], dtype=np.int64),
        )


class FakeDuckDBCatalog:
    """Capture hydrated chunk identifiers for verification."""

    calls: ClassVar[list[int]] = []

    def __init__(self, db_path: Path, vectors: Path) -> None:
        self.db_path = db_path
        self.vectors = vectors

    def __enter__(self) -> Self:
        """Enter the context manager while clearing recorded calls.

        Returns
        -------
        Self
            The current catalog instance.
        """
        FakeDuckDBCatalog.calls.clear()
        return self

    def __exit__(self, *_exc: object) -> None:  # pragma: no cover - trivial shim
        """Exit the context manager (no-op for tests)."""
        return

    @classmethod
    def get_chunk_by_id(cls, chunk_id: int) -> dict | None:
        """Return a mock chunk for the matching identifier.

        Returns
        -------
        dict | None
            Chunk metadata or ``None`` when the identifier does not match.
        """
        cls.calls.append(chunk_id)
        if chunk_id == MATCHING_CHUNK_ID:
            return {
                "uri": "src/demo.py",
                "start_line": 1,
                "end_line": 5,
                "preview": MATCHING_PREVIEW,
            }
        return None


def test_get_chunk_by_id_returns_single_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure DuckDBCatalog returns hydrated chunk metadata."""
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    parquet_path = vectors_dir / "chunks.parquet"
    _write_chunk_parquet(parquet_path)

    def _ensure_views_for_test(self: DuckDBCatalog) -> None:
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)

        parquet_pattern = str(self.vectors_dir / "**/*.parquet")
        safe_pattern = parquet_pattern.replace("'", "''")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW chunks AS SELECT * FROM read_parquet('{safe_pattern}')"  # noqa: S608
        )

    monkeypatch.setattr(DuckDBCatalog, "_ensure_views", _ensure_views_for_test, raising=False)

    catalog_path = tmp_path / "catalog.duckdb"
    with DuckDBCatalog(catalog_path, vectors_dir) as catalog:
        chunk = catalog.get_chunk_by_id(1)
        _expect(condition=chunk is not None, message="Expected chunk to be found")
        if chunk is not None:
            _expect(
                condition=chunk["uri"] == "src/example.py",
                message="Unexpected URI for chunk",
            )
            _expect(
                condition=chunk["preview"] == "Example preview",
                message="Unexpected preview text",
            )
        _expect(
            condition=catalog.get_chunk_by_id(9999) is None,
            message="Expected None for missing chunk",
        )


@pytest.mark.asyncio
async def test_semantic_search_handles_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify semantic search hydrates chunk metadata using DuckDB."""
    # Type ignore: SimpleNamespace used as module mock in tests
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace())  # type: ignore[arg-type]

    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    faiss_index = tmp_path / "index.faiss"
    faiss_index.write_bytes(b"faiss")
    duckdb_path = tmp_path / "catalog.duckdb"
    duckdb_path.write_bytes(b"")

    settings = DummySettings(
        faiss_index=faiss_index,
        duckdb_path=duckdb_path,
        vectors_dir=vectors_dir,
    )

    def _dummy_load_settings() -> DummySettings:
        return settings

    monkeypatch.setattr(semantic, "load_settings", _dummy_load_settings)
    monkeypatch.setattr(semantic, "VLLMClient", FakeVLLMClient)
    monkeypatch.setattr(semantic, "FAISSManager", FakeFAISSManager)
    monkeypatch.setattr(semantic, "DuckDBCatalog", FakeDuckDBCatalog)

    result = await semantic.semantic_search("test query", limit=1)

    findings = result.get("findings", [])
    _expect(condition=bool(findings), message="Expected findings to be non-empty")
    finding = findings[0]
    _expect(
        condition=finding.get("snippet") == MATCHING_PREVIEW,
        message="Unexpected snippet content",
    )
    _expect(
        condition=finding.get("location", {}).get("uri") == "src/demo.py",
        message="Unexpected hydrated URI",
    )
    _expect(
        condition=FakeDuckDBCatalog.calls == [MATCHING_CHUNK_ID],
        message="Unexpected chunk IDs during hydration",
    )
