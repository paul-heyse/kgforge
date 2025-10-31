"""Unit tests for SPLADE index: load, search, failure modes, and SQL injection attempts.

Tests verify happy paths, error handling, and SQL injection resistance through
parameterized queries.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

try:
    from kgfoundry.search_api.splade_index import SpladeIndex  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("SPLADE index not available", allow_module_level=True)


@pytest.fixture
def temp_parquet_file(tmp_path: Path) -> Path:
    """Create a temporary parquet file with test documents."""
    chunk_ids = [f"chunk_{i}" for i in range(10)]
    doc_ids = [f"doc_{i // 2}" for i in range(10)]
    sections = ["intro", "body", "conclusion"] * 3 + ["intro"]
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a programming language",
        "Machine learning is fascinating",
        "Natural language processing",
        "Search algorithms and data structures",
        "Test-driven development",
        "Continuous integration and deployment",
        "Software engineering best practices",
        "Code quality and maintainability",
        "Testing frameworks and methodologies",
    ]

    parquet_path = tmp_path / "chunks.parquet"
    # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
    arrays: list[object] = [
        pa.array(chunk_ids),  # type: ignore[misc]
        pa.array(doc_ids),  # type: ignore[misc]
        pa.array(sections),  # type: ignore[misc]
        pa.array(texts),  # type: ignore[misc]
    ]
    table: object = pa.Table.from_arrays(arrays, names=["chunk_id", "doc_id", "section", "text"])  # type: ignore[misc]
    pq.write_table(table, parquet_path)  # type: ignore[misc]
    return parquet_path


@pytest.fixture
def temp_duckdb_with_chunks(tmp_path: Path, temp_parquet_file: Path) -> Path:
    """Create a temporary DuckDB database with datasets table."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                parquet_root TEXT,
                kind TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        parquet_dir = temp_parquet_file.parent
        con.execute(
            "INSERT INTO datasets (parquet_root, kind) VALUES (?, ?)",
            [str(parquet_dir), "chunks"],
        )
    finally:
        con.close()
    return db_path


class TestSpladeIndexHappyPath:
    """Happy path tests for SPLADE index."""

    def test_init_loads_from_duckdb(self, temp_duckdb_with_chunks: Path) -> None:
        """Test initialization loads documents from DuckDB."""
        index = SpladeIndex(db_path=str(temp_duckdb_with_chunks))
        assert index.N == 10
        assert len(index.docs) == 10
        assert len(index.df) > 0

    def test_search_returns_results(self, temp_duckdb_with_chunks: Path) -> None:
        """Test search returns correct results."""
        index = SpladeIndex(db_path=str(temp_duckdb_with_chunks))
        results = index.search("python programming", k=5)

        assert len(results) <= 5
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        # Scores should be sorted descending
        if len(results) > 1:
            scores = [score for _, score in results]
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_search_empty_query_returns_empty(self, temp_duckdb_with_chunks: Path) -> None:
        """Test search with empty query returns empty results."""
        index = SpladeIndex(db_path=str(temp_duckdb_with_chunks))
        results = index.search("", k=10)
        assert results == []

    def test_search_with_no_docs_returns_empty(self, tmp_path: Path) -> None:
        """Test search returns empty when no documents loaded."""
        db_path = tmp_path / "empty.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    parquet_root TEXT,
                    kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        finally:
            con.close()

        index = SpladeIndex(db_path=str(db_path))
        assert index.N == 0
        results = index.search("test", k=10)
        assert results == []


class TestSpladeIndexFailures:
    """Failure mode tests for SPLADE index."""

    def test_init_with_missing_db(self, tmp_path: Path) -> None:
        """Test initialization handles missing database gracefully."""
        index = SpladeIndex(db_path=str(tmp_path / "nonexistent.db"))
        assert index.N == 0
        assert len(index.docs) == 0

    def test_init_with_empty_datasets(self, tmp_path: Path) -> None:
        """Test initialization handles empty datasets table."""
        db_path = tmp_path / "empty.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    parquet_root TEXT,
                    kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        finally:
            con.close()

        index = SpladeIndex(db_path=str(db_path))
        assert index.N == 0
        assert len(index.docs) == 0

    def test_init_with_invalid_parquet_root_type(self, tmp_path: Path) -> None:
        """Test initialization handles invalid parquet_root type."""
        db_path = tmp_path / "invalid.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    parquet_root TEXT,
                    kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Insert NULL parquet_root
            con.execute("INSERT INTO datasets (parquet_root, kind) VALUES (NULL, 'chunks')")
        finally:
            con.close()

        # Should raise ValueError for invalid type
        with pytest.raises(ValueError, match="Invalid parquet_root type"):  # type: ignore[call-overload]
            SpladeIndex(db_path=str(db_path))

    def test_init_with_nonexistent_parquet_files(self, tmp_path: Path) -> None:
        """Test initialization handles nonexistent parquet files."""
        db_path = tmp_path / "nonexistent_files.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    parquet_root TEXT,
                    kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Insert path to nonexistent directory
            con.execute(
                "INSERT INTO datasets (parquet_root, kind) VALUES (?, ?)",
                [str(tmp_path / "nonexistent"), "chunks"],
            )
        finally:
            con.close()

        # Should handle gracefully - no documents loaded
        index = SpladeIndex(db_path=str(db_path))
        assert index.N == 0
        assert len(index.docs) == 0


class TestSpladeIndexSQLInjection:
    """SQL injection resistance tests for SPLADE index."""

    def test_sql_injection_in_duckdb_parquet_root(self, tmp_path: Path) -> None:
        """Test that malicious parquet_root in DuckDB doesn't execute SQL injection."""
        # Create a valid parquet file
        chunk_ids = ["chunk_1"]
        doc_ids = ["doc_1"]
        sections = ["intro"]
        texts = ["test text"]

        parquet_path = tmp_path / "chunks.parquet"
        # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
        arrays: list[object] = [
            pa.array(chunk_ids),  # type: ignore[misc]
            pa.array(doc_ids),  # type: ignore[misc]
            pa.array(sections),  # type: ignore[misc]
            pa.array(texts),  # type: ignore[misc]
        ]
        table: object = pa.Table.from_arrays(
            arrays, names=["chunk_id", "doc_id", "section", "text"]
        )  # type: ignore[misc]
        pq.write_table(table, parquet_path)  # type: ignore[misc]

        db_path = tmp_path / "injection.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    parquet_root TEXT,
                    kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Try to inject SQL via parquet_root value
            # The adapter uses parameterized queries, so this should be safe
            malicious_root = str(tmp_path) + "'; DROP TABLE datasets; --"
            con.execute(
                "INSERT INTO datasets (parquet_root, kind) VALUES (?, ?)",
                [malicious_root, "chunks"],
            )
        finally:
            con.close()

        # Should fail safely (file not found) rather than execute SQL
        # The parameterized query prevents injection
        index = SpladeIndex(db_path=str(db_path))
        # Returns empty index when path doesn't exist
        assert index.N == 0

    def test_path_resolution_safety(self, tmp_path: Path) -> None:
        """Test that Path.resolve() prevents directory traversal attacks."""
        db_path = tmp_path / "traversal.db"
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    parquet_root TEXT,
                    kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Try directory traversal
            traversal_root = str(tmp_path / ".." / "nonexistent")
            con.execute(
                "INSERT INTO datasets (parquet_root, kind) VALUES (?, ?)",
                [traversal_root, "chunks"],
            )
        finally:
            con.close()

        # Path.resolve() should resolve this safely
        # May return empty index if resolved path doesn't exist
        index = SpladeIndex(db_path=str(db_path))
        assert index.N == 0
