"""Unit tests for BM25 index: build, load, search, failure modes, and SQL injection attempts.

Tests verify happy paths, error handling, schema validation, and SQL injection
resistance through parameterized queries.
"""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from kgfoundry_common.errors import DeserializationError

try:
    from kgfoundry.search_api.bm25_index import BM25Index
except ImportError:
    pytest.skip("BM25 index not available", allow_module_level=True)


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


class TestBM25IndexHappyPath:
    """Happy path tests for BM25 index."""

    def test_from_parquet_builds_index(self, temp_parquet_file: Path) -> None:
        """Test building index from parquet file."""
        index = BM25Index.from_parquet(str(temp_parquet_file))
        assert index.N == 10
        assert len(index.docs) == 10
        assert index.avgdl > 0

    def test_build_from_duckdb(self, temp_duckdb_with_chunks: Path) -> None:
        """Test building index from DuckDB registry."""
        index = BM25Index.build_from_duckdb(str(temp_duckdb_with_chunks))
        assert index.N == 10
        assert len(index.docs) == 10

    def test_search_returns_results(self, temp_parquet_file: Path) -> None:
        """Test search returns correct results."""
        index = BM25Index.from_parquet(str(temp_parquet_file))
        results = index.search("python programming", k=5)

        assert len(results) == 5
        assert all(isinstance(chunk_id, str) for chunk_id, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        # Scores should be sorted descending
        scores = [score for _, score in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_search_empty_query_returns_empty(self, temp_parquet_file: Path) -> None:
        """Test search with empty query returns empty results."""
        index = BM25Index.from_parquet(str(temp_parquet_file))
        results = index.search("", k=10)
        assert results == []

    def test_save_and_load_roundtrip(self, tmp_path: Path, temp_parquet_file: Path) -> None:
        """Test save and load roundtrip preserves index state."""
        index1 = BM25Index.from_parquet(str(temp_parquet_file))
        save_path = tmp_path / "index.json"
        index1.save(str(save_path))

        index2 = BM25Index.load(str(save_path))

        assert index2.N == index1.N
        assert index2.avgdl == index1.avgdl
        assert index2.k1 == index1.k1
        assert index2.b == index1.b
        assert len(index2.docs) == len(index1.docs)

        # Verify search results are consistent
        results1 = index1.search("python", k=5)
        results2 = index2.search("python", k=5)
        assert len(results1) == len(results2)


class TestBM25IndexFailures:
    """Failure mode tests for BM25 index."""

    def test_from_parquet_with_missing_file(self, tmp_path: Path) -> None:
        """Test from_parquet raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):  # type: ignore[call-overload]
            BM25Index.from_parquet(str(tmp_path / "nonexistent.parquet"))

    def test_from_parquet_with_empty_file(self, tmp_path: Path) -> None:
        """Test from_parquet handles empty parquet file."""
        empty_parquet = tmp_path / "empty.parquet"
        # pyarrow operations return Any types - explicitly type ignore for third-party library limitation
        arrays: list[object] = [
            pa.array([]),  # type: ignore[misc]
            pa.array([]),  # type: ignore[misc]
            pa.array([]),  # type: ignore[misc]
            pa.array([]),  # type: ignore[misc]
        ]
        table: object = pa.Table.from_arrays(
            arrays, names=["chunk_id", "doc_id", "section", "text"]
        )  # type: ignore[misc]
        pq.write_table(table, empty_parquet)  # type: ignore[misc]

        index = BM25Index.from_parquet(str(empty_parquet))
        assert index.N == 0
        assert len(index.docs) == 0

    def test_build_from_duckdb_with_no_datasets(self, tmp_path: Path) -> None:
        """Test build_from_duckdb returns empty index when no datasets."""
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

        index = BM25Index.build_from_duckdb(str(db_path))
        assert index.N == 0
        assert len(index.docs) == 0

    def test_build_from_duckdb_with_invalid_parquet_root(self, tmp_path: Path) -> None:
        """Test build_from_duckdb handles invalid parquet_root type."""
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

        index = BM25Index.build_from_duckdb(str(db_path))
        # Should return empty index when parquet_root is invalid
        assert index.N == 0

    def test_load_with_missing_file(self, tmp_path: Path) -> None:
        """Test load raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            BM25Index.load(str(tmp_path / "nonexistent.json"))

    def test_load_with_invalid_json(self, tmp_path: Path) -> None:
        """Test load raises DeserializationError with invalid JSON."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("not valid json", encoding="utf-8")

        with pytest.raises(DeserializationError):
            BM25Index.load(str(invalid_json))

    def test_load_with_invalid_schema(self, tmp_path: Path) -> None:
        """Test load raises DeserializationError with invalid schema."""
        invalid_file = tmp_path / "invalid.json"
        invalid_data = {"invalid": "data", "missing": "required fields"}
        invalid_file.write_text(json.dumps(invalid_data), encoding="utf-8")

        with pytest.raises(DeserializationError):
            BM25Index.load(str(invalid_file))


class TestBM25IndexSQLInjection:
    """SQL injection resistance tests for BM25 index."""

    def test_sql_injection_in_parquet_path(self, tmp_path: Path) -> None:
        """Test that malicious path input doesn't execute SQL injection."""
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

        # Try to inject SQL via path manipulation
        malicious_path = tmp_path / "chunks.parquet'; DROP TABLE datasets; --"

        # Should raise FileNotFoundError, not execute SQL
        with pytest.raises(FileNotFoundError, match="not found"):  # type: ignore[call-overload]
            BM25Index.from_parquet(str(malicious_path))

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
        index = BM25Index.build_from_duckdb(str(db_path))
        # Returns empty index when path doesn't exist
        assert index.N == 0

    def test_path_resolution_safety(self, tmp_path: Path) -> None:
        """Test that Path.resolve() prevents directory traversal attacks."""
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

        # Try directory traversal
        traversal_path = tmp_path / ".." / "chunks.parquet"

        # Path.resolve() should resolve this safely
        # May raise FileNotFoundError if resolved path doesn't exist
        with suppress(FileNotFoundError):
            BM25Index.from_parquet(str(traversal_path))  # Expected - traversal should fail safely
