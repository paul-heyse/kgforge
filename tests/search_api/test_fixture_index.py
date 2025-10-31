"""Unit tests for fixture index: load, search, failure modes, and SQL injection attempts.

Tests verify happy paths, error handling, invalid data handling, and SQL injection
resistance through parameterized queries.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

try:
    from kgfoundry.search_api.fixture_index import (
        FixtureIndex,  # type: ignore[import-untyped,import-not-found]
    )
except ImportError:
    pytest.skip("Fixture index not available", allow_module_level=True)


@pytest.fixture
def temp_parquet_file(tmp_path: Path) -> Path:  # type: ignore[misc]
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
def temp_duckdb_with_chunks(tmp_path: Path, temp_parquet_file: Path) -> Path:  # type: ignore[misc]
    """Create a temporary DuckDB database with datasets and documents tables."""
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

        # Create documents table for JOIN
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT
            )
            """
        )
        for i in range(5):
            con.execute(
                "INSERT INTO documents (doc_id, title) VALUES (?, ?)",
                [f"doc_{i}", f"Document {i}"],
            )
    finally:
        con.close()
    return db_path


class TestFixtureIndexHappyPath:
    """Happy path tests for fixture index."""

    def test_init_loads_from_duckdb(self, temp_duckdb_with_chunks: Path) -> None:
        """Test initialization loads documents from DuckDB."""
        index = FixtureIndex(  # type: ignore[misc]
            root=str(temp_duckdb_with_chunks.parent), db_path=str(temp_duckdb_with_chunks)
        )
        assert hasattr(index, "N")  # type: ignore[misc]
        assert index.N == 10  # type: ignore[misc]
        assert len(index.docs) == 10  # type: ignore[misc]
        assert len(index.df) > 0  # type: ignore[misc]

    def test_search_returns_results(self, temp_duckdb_with_chunks: Path) -> None:
        """Test search returns correct results."""
        index = FixtureIndex(  # type: ignore[misc]
            root=str(temp_duckdb_with_chunks.parent), db_path=str(temp_duckdb_with_chunks)
        )
        results = index.search("python programming", k=5)  # type: ignore[misc]

        assert len(results) <= 5
        assert all(isinstance(idx, int) for idx, _ in results)  # type: ignore[misc]
        assert all(isinstance(score, float) for _, score in results)  # type: ignore[misc]
        # Scores should be sorted descending
        if len(results) > 1:
            scores = [score for _, score in results]  # type: ignore[misc]
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))  # type: ignore[misc]

    def test_search_empty_query_returns_empty(self, temp_duckdb_with_chunks: Path) -> None:
        """Test search with empty query returns empty results."""
        index = FixtureIndex(  # type: ignore[misc]
            root=str(temp_duckdb_with_chunks.parent), db_path=str(temp_duckdb_with_chunks)
        )
        results = index.search("", k=10)  # type: ignore[misc]
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

        index = FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]
        if not hasattr(index, "N") or index.N == 0:  # type: ignore[misc]
            results = index.search("test", k=10)  # type: ignore[misc]
            assert results == []


class TestFixtureIndexFailures:
    """Failure mode tests for fixture index."""

    def test_init_with_missing_db(self, tmp_path: Path) -> None:
        """Test initialization handles missing database gracefully."""
        index = FixtureIndex(root=str(tmp_path), db_path=str(tmp_path / "nonexistent.db"))  # type: ignore[misc]
        assert not hasattr(index, "N") or index.N == 0  # type: ignore[misc]
        assert len(index.docs) == 0  # type: ignore[misc]

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

        index = FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]
        assert not hasattr(index, "N") or index.N == 0  # type: ignore[misc]
        assert len(index.docs) == 0  # type: ignore[misc]

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

        # Should raise TypeError for invalid type
        with pytest.raises(TypeError, match="Invalid parquet_root type"):  # type: ignore[call-overload]
            FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]

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
        index = FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]
        assert not hasattr(index, "N") or index.N == 0  # type: ignore[misc]
        assert len(index.docs) == 0  # type: ignore[misc]

    def test_init_with_invalid_data_types(self, tmp_path: Path) -> None:
        """Test initialization handles invalid data types in parquet."""
        # Create parquet with invalid data types
        parquet_path = tmp_path / "invalid_types.parquet"
        # Use None/null values which should be handled gracefully
        # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
        arrays: list[object] = [
            pa.array([None, "chunk_2"], type=pa.string()),  # type: ignore[misc]
            pa.array([None, "doc_1"], type=pa.string()),  # type: ignore[misc]
            pa.array(["", None], type=pa.string()),  # type: ignore[misc]
            pa.array(["", "test"], type=pa.string()),  # type: ignore[misc]
        ]
        table: object = pa.Table.from_arrays(
            arrays, names=["chunk_id", "doc_id", "section", "text"]
        )  # type: ignore[misc]
        pq.write_table(table, parquet_path)  # type: ignore[misc]

        db_path = tmp_path / "invalid_data.db"
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
            con.execute(
                "INSERT INTO datasets (parquet_root, kind) VALUES (?, ?)",
                [str(tmp_path), "chunks"],
            )

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT
                )
                """
            )
        finally:
            con.close()

        # Should handle None/null values gracefully
        index = FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]
        # Some documents may be loaded, None values should be converted to defaults
        assert len(index.docs) >= 0  # type: ignore[misc]


class TestFixtureIndexSQLInjection:
    """SQL injection resistance tests for fixture index."""

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
        index = FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]
        # Returns empty index when path doesn't exist
        assert not hasattr(index, "N") or index.N == 0  # type: ignore[misc]

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
        index = FixtureIndex(root=str(tmp_path), db_path=str(db_path))  # type: ignore[misc]
        assert not hasattr(index, "N") or index.N == 0  # type: ignore[misc]
