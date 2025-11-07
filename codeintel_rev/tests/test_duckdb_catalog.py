"""Tests covering DuckDB catalog helpers and semantic search hydration."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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


def _write_chunk_parquet_with_embedding(parquet_path: Path, vec_dim: int) -> None:
    """Write a parquet file containing an embedding column for tests.

    Parameters
    ----------
    parquet_path : Path
        Destination parquet path.
    vec_dim : int
        Embedding dimension.

    Returns
    -------
    None
    """
    values = ", ".join(f"{0.1 * (i + 1):.1f}" for i in range(vec_dim))
    conn = duckdb.connect()
    try:
        conn.execute(
            f"""
            CREATE TABLE chunks AS
            SELECT
                1 AS id,
                'src/example.py' AS uri,
                0 AS start_line,
                0 AS end_line,
                0 AS start_byte,
                10 AS end_byte,
                'Example preview' AS preview,
                LIST_VALUE({values})::FLOAT[] AS embedding
            """
        )
        conn.execute(
            "COPY chunks TO ? (FORMAT PARQUET)",
            [str(parquet_path)],
        )
    finally:
        conn.close()


class FakeVLLMClient:
    """Record embedding requests made during semantic search."""

    def __init__(self, _config: object) -> None:
        self.calls: list[str] = []

    def embed_single(self, query: str) -> list[float]:
        """Return a deterministic embedding vector for *query*.

        Parameters
        ----------
        query : str
            Query text to embed.

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
        self.gpu_index = None

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

        Parameters
        ----------
        _query_vec : np.ndarray
            Query vector (unused in mock).
        k : int
            Number of results to return.

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


class RecordingCatalog:
    """Capture hydrated chunk identifiers for verification."""

    def __init__(self) -> None:
        """Initialise the in-memory catalog."""
        self.queries: list[list[int]] = []

    def query_by_ids(self, ids: list[int]) -> list[dict]:
        """Return chunk metadata for any matching identifiers.

        Parameters
        ----------
        ids : list[int]
            Chunk identifiers to query.

        Returns
        -------
        list[dict]
            Chunk metadata records matching the provided identifiers.
        """
        self.queries.append(list(ids))
        if MATCHING_CHUNK_ID in ids:
            return [
                {
                    "id": MATCHING_CHUNK_ID,
                    "uri": "src/demo.py",
                    "start_line": 1,
                    "end_line": 5,
                    "preview": MATCHING_PREVIEW,
                }
            ]
        return []


class StubContext:
    """Stub service context yielding deterministic dependencies."""

    def __init__(self) -> None:
        """Initialise stubbed dependencies for semantic search tests."""
        self.faiss_manager = FakeFAISSManager()
        self.vllm_client = FakeVLLMClient(SimpleNamespace())
        self.settings = SimpleNamespace()
        self._limits: list[str] = []
        self._error: str | None = None
        self.catalog = RecordingCatalog()

    def ensure_faiss_ready(self) -> tuple[bool, list[str], str | None]:
        """Return readiness tuple.

        Returns
        -------
        tuple[bool, list[str], str | None]
            Tuple of (ready, limits, error).
        """
        return True, list(self._limits), self._error

    @contextmanager
    def open_catalog(self) -> Iterator[RecordingCatalog]:
        """Yield stub catalog.

        Yields
        ------
        RecordingCatalog
            Stub catalog instance.
        """
        yield self.catalog


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


def test_get_embeddings_by_ids_returns_correct_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure embedding queries return matrices with consistent shape."""
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    parquet_path = vectors_dir / "chunks.parquet"
    _write_chunk_parquet_with_embedding(parquet_path, vec_dim=3)

    def _ensure_views_for_test(self: DuckDBCatalog) -> None:
        if not self.conn:
            msg = "Connection not open"
            raise RuntimeError(msg)
        parquet_pattern = str(self.vectors_dir / "**/*.parquet")
        self.conn.execute(
            "CREATE OR REPLACE VIEW chunks AS SELECT * FROM read_parquet(?)",
            [parquet_pattern],
        )

    monkeypatch.setattr(DuckDBCatalog, "_ensure_views", _ensure_views_for_test, raising=False)

    catalog_path = tmp_path / "catalog.duckdb"
    with DuckDBCatalog(catalog_path, vectors_dir) as catalog:
        empty_matrix = catalog.get_embeddings_by_ids([])
        _expect(condition=empty_matrix.shape == (0, 3), message="Expected (0, 3) for empty query")

        result = catalog.get_embeddings_by_ids([1])
        _expect(condition=result.shape == (1, 3), message="Expected single embedding row")
        _expect(
            condition=np.allclose(result[0], [0.1, 0.2, 0.3]), message="Unexpected embedding values"
        )

        missing = catalog.get_embeddings_by_ids([999])
        _expect(condition=missing.shape == (0, 3), message="Missing IDs should return empty matrix")


@pytest.mark.asyncio
async def test_semantic_search_handles_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify semantic search hydrates chunk metadata using DuckDB."""
    # Type ignore: SimpleNamespace used as module mock in tests
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace())  # type: ignore[arg-type]
    _ = tmp_path

    context = StubContext()

    monkeypatch.setattr(semantic, "get_service_context", lambda: context)

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
        condition=context.catalog.queries == [[MATCHING_CHUNK_ID]],
        message="Unexpected chunk IDs during hydration",
    )
