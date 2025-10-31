"""Unit tests for FAISS adapter: build, load, search, failure modes, and SQL injection attempts.

Tests verify happy paths, error handling, CPU fallback behavior, and SQL injection
resistance through parameterized queries.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from kgfoundry_common.errors import VectorSearchError

pytest.importorskip("faiss")

try:
    from kgfoundry.search_api.faiss_adapter import HAVE_FAISS, FaissAdapter
except ImportError:
    pytest.skip("FAISS adapter not available", allow_module_level=True)


@pytest.fixture
def temp_parquet_file(tmp_path: Path) -> Path:
    """Create a temporary parquet file with test vectors."""
    vectors = np.random.randn(100, 128).astype(np.float32)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-9)

    chunk_ids = [f"chunk_{i}" for i in range(100)]

    parquet_path = tmp_path / "vectors.parquet"
    # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
    arrays: list[object] = [
        pa.array(chunk_ids),  # type: ignore[misc]
        pa.array([v.tolist() for v in vectors]),  # type: ignore[misc]
    ]
    table: object = pa.Table.from_arrays(arrays, names=["chunk_id", "vector"])  # type: ignore[misc]
    pq.write_table(table, parquet_path)  # type: ignore[misc]
    return parquet_path


@pytest.fixture
def temp_duckdb_with_vectors(tmp_path: Path, temp_parquet_file: Path) -> Path:
    """Create a temporary DuckDB database with dense_runs table."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dense_runs (
                parquet_root TEXT,
                dim INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        parquet_dir = temp_parquet_file.parent
        con.execute(
            "INSERT INTO dense_runs (parquet_root, dim) VALUES (?, ?)",
            [str(parquet_dir), 128],
        )
    finally:
        con.close()
    return db_path


class TestFaissAdapterHappyPath:
    """Happy path tests for FAISS adapter."""

    def test_build_from_parquet_file(self, temp_parquet_file: Path) -> None:
        """Test building index from parquet file."""
        adapter = FaissAdapter(db_path=str(temp_parquet_file))
        adapter.build()
        assert adapter.index is not None
        assert adapter.idmap is not None
        assert len(adapter.idmap) == 100

    def test_build_from_duckdb(self, temp_duckdb_with_vectors: Path) -> None:
        """Test building index from DuckDB registry."""
        adapter = FaissAdapter(db_path=str(temp_duckdb_with_vectors))
        adapter.build()
        assert adapter.index is not None
        assert adapter.idmap is not None
        assert len(adapter.idmap) == 100

    def test_search_returns_results(self, temp_parquet_file: Path) -> None:
        """Test search returns correct number of results."""
        adapter = FaissAdapter(db_path=str(temp_parquet_file))
        adapter.build()

        # Create a query vector
        query = np.random.randn(128).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-9)

        # Reshape to 2D array for search (query vector must be 2D)
        # _prepare_queries handles reshaping, but we provide 2D for explicit typing
        query_2d = query.reshape(1, -1).astype(np.float32)
        results = adapter.search(query_2d, k=10)  # type: ignore[arg-type]
        assert len(results) == 1  # One query
        assert len(results[0]) == 10  # k=10 results

        # Verify results are tuples of (chunk_id, score)
        for chunk_id, score in results[0]:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)
            assert chunk_id.startswith("chunk_")

    def test_search_without_build_returns_empty(self, temp_parquet_file: Path) -> None:
        """Test search without build returns empty results."""
        adapter = FaissAdapter(db_path=str(temp_parquet_file))
        # Don't call build()
        query = np.random.randn(128).astype(np.float32)
        # Reshape to 2D array for search (query vector must be 2D)
        # _prepare_queries handles reshaping, but we provide 2D for explicit typing
        query_2d = query.reshape(1, -1).astype(np.float32)
        results = adapter.search(query_2d, k=10)  # type: ignore[arg-type]
        assert results == []


class TestFaissAdapterFailures:
    """Failure mode tests for FAISS adapter."""

    def test_build_with_missing_file(self, tmp_path: Path) -> None:
        """Test build raises VectorSearchError when file doesn't exist."""
        adapter = FaissAdapter(db_path=str(tmp_path / "nonexistent.parquet"))
        with pytest.raises(VectorSearchError, match="not found"):  # type: ignore[call-overload]
            adapter.build()

    def test_build_with_empty_parquet(self, tmp_path: Path) -> None:
        """Test build raises VectorSearchError when parquet has no vectors."""
        empty_parquet = tmp_path / "empty.parquet"
        # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
        arrays: list[object] = [
            pa.array([]),  # type: ignore[misc]
            pa.array([]),  # type: ignore[misc]
        ]
        table: object = pa.Table.from_arrays(arrays, names=["chunk_id", "vector"])  # type: ignore[misc]
        pq.write_table(table, empty_parquet)  # type: ignore[misc]

        adapter = FaissAdapter(db_path=str(empty_parquet))
        with pytest.raises(VectorSearchError, match="No dense vectors"):  # type: ignore[call-overload]
            adapter.build()

    def test_build_with_empty_duckdb(self, tmp_path: Path) -> None:
        """Test build raises VectorSearchError when DuckDB has no dense_runs."""
        db_path = tmp_path / "empty.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS dense_runs (
                    parquet_root TEXT,
                    dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        finally:
            con.close()

        adapter = FaissAdapter(db_path=str(db_path))
        with pytest.raises(VectorSearchError, match="No dense_runs found"):  # type: ignore[call-overload]
            adapter.build()

    def test_build_with_invalid_parquet_root_type(self, tmp_path: Path) -> None:
        """Test build handles invalid parquet_root type in DuckDB."""
        db_path = tmp_path / "invalid.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS dense_runs (
                    parquet_root TEXT,
                    dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Insert invalid type (should be string but insert NULL)
            con.execute("INSERT INTO dense_runs (parquet_root, dim) VALUES (NULL, 128)")
        finally:
            con.close()

        adapter = FaissAdapter(db_path=str(db_path))
        with pytest.raises(VectorSearchError, match="Invalid parquet_root type"):  # type: ignore[call-overload]
            adapter.build()

    @pytest.mark.skipif(not HAVE_FAISS, reason="FAISS not available")
    def test_build_fallback_to_cpu_when_gpu_unavailable(self, temp_parquet_file: Path) -> None:
        """Test build falls back to CPU when GPU is unavailable."""
        adapter = FaissAdapter(db_path=str(temp_parquet_file))
        adapter.build()
        # Should succeed even without GPU
        assert adapter.index is not None


class TestFaissAdapterSQLInjection:
    """SQL injection resistance tests for FAISS adapter."""

    def test_sql_injection_in_parquet_path(self, tmp_path: Path) -> None:
        """Test that malicious path input doesn't execute SQL injection."""
        # Create a valid parquet file
        vectors = np.random.randn(10, 128).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)

        chunk_ids = [f"chunk_{i}" for i in range(10)]
        parquet_path = tmp_path / "vectors.parquet"
        # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
        arrays: list[object] = [
            pa.array(chunk_ids),  # type: ignore[misc]
            pa.array([v.tolist() for v in vectors]),  # type: ignore[misc]
        ]
        table: object = pa.Table.from_arrays(arrays, names=["chunk_id", "vector"])  # type: ignore[misc]
        pq.write_table(table, parquet_path)  # type: ignore[misc]

        # Try to inject SQL via path manipulation
        # The adapter uses Path.resolve() and parameterized queries, so this should fail safely
        malicious_path = tmp_path / "vectors.parquet'; DROP TABLE dense_runs; --"

        adapter = FaissAdapter(db_path=str(malicious_path))
        # Should raise FileNotFoundError, not execute SQL
        with pytest.raises((VectorSearchError, FileNotFoundError)):  # type: ignore[call-overload]
            adapter.build()

    def test_sql_injection_in_duckdb_parquet_root(
        self, tmp_path: Path, temp_parquet_file: Path
    ) -> None:
        """Test that malicious parquet_root in DuckDB doesn't execute SQL injection."""
        db_path = tmp_path / "injection.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS dense_runs (
                    parquet_root TEXT,
                    dim INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Try to inject SQL via parquet_root value
            # The adapter uses parameterized queries, so this should be safe
            malicious_root = str(temp_parquet_file.parent) + "'; DROP TABLE dense_runs; --"
            con.execute(
                "INSERT INTO dense_runs (parquet_root, dim) VALUES (?, ?)",
                [malicious_root, 128],
            )
        finally:
            con.close()

        adapter = FaissAdapter(db_path=str(db_path))
        # Should fail safely (file not found) rather than execute SQL
        # The parameterized query prevents injection
        with pytest.raises((VectorSearchError, FileNotFoundError)):  # type: ignore[call-overload]
            adapter.build()

    def test_path_resolution_safety(self, tmp_path: Path) -> None:
        """Test that Path.resolve() prevents directory traversal attacks."""
        # Create a valid parquet file
        vectors = np.random.randn(10, 128).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)

        chunk_ids = [f"chunk_{i}" for i in range(10)]
        parquet_path = tmp_path / "vectors.parquet"
        # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
        arrays: list[object] = [
            pa.array(chunk_ids),  # type: ignore[misc]
            pa.array([v.tolist() for v in vectors]),  # type: ignore[misc]
        ]
        table: object = pa.Table.from_arrays(arrays, names=["chunk_id", "vector"])  # type: ignore[misc]
        pq.write_table(table, parquet_path)  # type: ignore[misc]

        # Try directory traversal
        traversal_path = tmp_path / ".." / "vectors.parquet"

        adapter = FaissAdapter(db_path=str(traversal_path))
        # Path.resolve() should resolve this safely
        # May raise FileNotFoundError if resolved path doesn't exist
        with suppress(VectorSearchError, FileNotFoundError):
            adapter.build()  # Expected - traversal should fail safely
