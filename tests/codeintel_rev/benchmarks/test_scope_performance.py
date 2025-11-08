"""Performance benchmarks for scope filtering operations.

This module benchmarks the performance overhead of scope filtering compared to
baseline queries without filtering. Tests verify that filtering adds <5ms overhead
for typical query sizes (1000 chunks).

Benchmarks:
- Baseline: query_by_ids (no filtering)
- Language filter: query_by_filters with languages=["python"]
- Path globs: query_by_filters with include_globs=["src/**"]
- Combined filters: query_by_filters with both language and path filters

All benchmarks use pytest-benchmark plugin for accurate timing and reporting.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from codeintel_rev.io.duckdb_catalog import DuckDBCatalog

# Number of chunks to create for benchmark
BENCHMARK_CHUNK_COUNT = 100_000

# Number of chunks to query in each benchmark
QUERY_CHUNK_COUNT = 1000

# Performance threshold: filtering overhead should be <5ms
MAX_FILTER_OVERHEAD_MS = 5.0


def _generate_chunk_uri(
    chunk_id: int, languages: list[str], directories: list[str], extensions: dict[str, list[str]]
) -> str:
    """Generate a URI for a chunk based on its ID.

    Parameters
    ----------
    chunk_id : int
        Chunk identifier.
    languages : list[str]
        Available languages.
    directories : list[str]
        Available directories.
    extensions : dict[str, list[str]]
        Language to extension mapping.

    Returns
    -------
    str
        Generated URI for the chunk.
    """
    lang = languages[chunk_id % len(languages)]
    directory = directories[chunk_id % len(directories)]
    ext = extensions[lang][chunk_id % len(extensions[lang])]

    # Create nested paths occasionally
    if chunk_id % 10 == 0:
        return f"{directory}/nested/deep/file{chunk_id}{ext}"
    if chunk_id % 5 == 0:
        return f"{directory}/subdir/file{chunk_id}{ext}"
    return f"{directory}/file{chunk_id}{ext}"


def _generate_chunk_data(
    chunk_id: int, uri: str
) -> tuple[int, str, int, int, int, int, str, list[float]]:
    """Generate chunk data tuple for a given chunk ID and URI.

    Parameters
    ----------
    chunk_id : int
        Chunk identifier.
    uri : str
        Chunk URI.

    Returns
    -------
    tuple[int, str, int, int, int, int, str, list[float]]
        Chunk data tuple (id, uri, start_line, end_line, start_byte, end_byte, preview, embedding).
    """
    start_line = (chunk_id % 100) + 1
    end_line = start_line + 10
    start_byte = chunk_id * 100
    end_byte = start_byte + 500
    preview = f"Code chunk {chunk_id} in {uri}"
    # Use a simpler embedding representation for performance
    embedding = [float(x % 100) / 100.0 for x in range(2560)]  # 2560-dim embedding

    return (chunk_id, uri, start_line, end_line, start_byte, end_byte, preview, embedding)


@pytest.fixture
def benchmark_catalog(tmp_path: Path) -> DuckDBCatalog:
    """Create a DuckDB catalog with 100K chunks for benchmarking.

    Creates a catalog with diverse URIs representing various languages and
    directory structures to simulate real-world codebase indexing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for DuckDB database file.

    Returns
    -------
    DuckDBCatalog
        Catalog instance with 100K chunks loaded.

    Notes
    -----
    This fixture creates a large dataset (100K chunks) which may take several
    seconds to initialize. The fixture is session-scoped to avoid recreating
    it for each benchmark test.
    """
    db_path = tmp_path / "benchmark.duckdb"
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()

    catalog = DuckDBCatalog(db_path, vectors_dir)
    catalog.conn = duckdb.connect(str(db_path))

    # Generate diverse URIs for benchmarking
    # Mix of languages: Python, TypeScript, JavaScript, Rust, Go, Java, etc.
    # Mix of directories: src/, tests/, lib/, docs/, etc.
    languages = ["python", "typescript", "javascript", "rust", "go", "java"]
    directories = ["src", "tests", "lib", "docs", "scripts", "config"]
    extensions = {
        "python": [".py", ".pyi"],
        "typescript": [".ts", ".tsx"],
        "javascript": [".js", ".jsx"],
        "rust": [".rs"],
        "go": [".go"],
        "java": [".java"],
    }

    # Create chunks table structure first
    catalog.conn.execute(
        """
        CREATE TABLE chunks (
            id BIGINT,
            uri VARCHAR,
            start_line INTEGER,
            end_line INTEGER,
            start_byte BIGINT,
            end_byte BIGINT,
            preview VARCHAR,
            embedding FLOAT[]
        )
        """
    )

    # Generate and insert chunks in batches for performance
    batch_size = 10_000
    chunk_id = 1

    for batch_start in range(0, BENCHMARK_CHUNK_COUNT, batch_size):
        batch_end = min(batch_start + batch_size, BENCHMARK_CHUNK_COUNT)
        batch_data: list[tuple[int, str, int, int, int, int, str, list[float]]] = []

        for _ in range(batch_start, batch_end):
            uri = _generate_chunk_uri(chunk_id, languages, directories, extensions)
            chunk_data = _generate_chunk_data(chunk_id, uri)
            batch_data.append(chunk_data)
            chunk_id += 1

        # Insert batch using executemany for better performance
        catalog.conn.executemany(
            """
            INSERT INTO chunks (id, uri, start_line, end_line, start_byte, end_byte, preview, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch_data,
        )

    # Create index on uri column for efficient filtering
    catalog.conn.execute("CREATE INDEX IF NOT EXISTS idx_uri ON chunks(uri)")

    return catalog


@pytest.mark.benchmark
class TestScopeFilteringPerformance:
    """Benchmark scope filtering performance overhead."""

    def test_baseline_query_by_ids(self, benchmark_catalog: DuckDBCatalog, benchmark) -> None:
        """Benchmark baseline query_by_ids (no filtering).

        This establishes the baseline performance for retrieving chunks by ID
        without any filtering. All other benchmarks compare against this.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        # Select random chunk IDs for querying
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        def _query() -> list[dict]:
            return benchmark_catalog.query_by_ids(chunk_ids)

        result = benchmark(_query)
        assert len(result) == QUERY_CHUNK_COUNT
        assert all("id" in chunk for chunk in result)

    def test_language_filter_performance(self, benchmark_catalog: DuckDBCatalog, benchmark) -> None:
        """Benchmark query_by_filters with language filter.

        Measures the overhead of filtering chunks by programming language
        (Python files only). Expected overhead: <5ms compared to baseline.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        def _query() -> list[dict]:
            return benchmark_catalog.query_by_filters(chunk_ids, languages=["python"])

        result = benchmark(_query)
        # Verify filtering worked (should return only Python files)
        assert len(result) < QUERY_CHUNK_COUNT  # Some chunks filtered out
        assert all(chunk["uri"].endswith((".py", ".pyi")) for chunk in result)

    def test_path_glob_filter_performance(
        self, benchmark_catalog: DuckDBCatalog, benchmark
    ) -> None:
        """Benchmark query_by_filters with path glob pattern.

        Measures the overhead of filtering chunks by path pattern (src/**).
        Expected overhead: <5ms compared to baseline.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        def _query() -> list[dict]:
            return benchmark_catalog.query_by_filters(chunk_ids, include_globs=["src/**"])

        result = benchmark(_query)
        # Verify filtering worked (should return only src/ paths)
        assert len(result) < QUERY_CHUNK_COUNT  # Some chunks filtered out
        assert all(chunk["uri"].startswith("src/") for chunk in result)

    def test_combined_filters_performance(
        self, benchmark_catalog: DuckDBCatalog, benchmark
    ) -> None:
        """Benchmark query_by_filters with combined language and path filters.

        Measures the overhead of filtering chunks by both language and path
        patterns simultaneously. Expected overhead: <5ms compared to baseline.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        def _query() -> list[dict]:
            return benchmark_catalog.query_by_filters(
                chunk_ids,
                include_globs=["src/**"],
                languages=["python"],
            )

        result = benchmark(_query)
        # Verify filtering worked (should return only Python files in src/)
        assert len(result) < QUERY_CHUNK_COUNT  # Some chunks filtered out
        assert all(
            chunk["uri"].startswith("src/") and chunk["uri"].endswith((".py", ".pyi"))
            for chunk in result
        )

    def test_complex_glob_filter_performance(
        self, benchmark_catalog: DuckDBCatalog, benchmark
    ) -> None:
        """Benchmark query_by_filters with complex glob pattern.

        Measures the overhead of filtering chunks using complex glob patterns
        that require Python post-filtering (e.g., src/**/test_*.py).
        Expected overhead: <10ms (higher than simple globs due to Python filtering).

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        def _query() -> list[dict]:
            return benchmark_catalog.query_by_filters(chunk_ids, include_globs=["src/**/test_*.py"])

        result = benchmark(_query)
        # Verify filtering worked (should return only test files in src/)
        assert len(result) < QUERY_CHUNK_COUNT  # Some chunks filtered out
        assert all(
            chunk["uri"].startswith("src/")
            and "test_" in chunk["uri"]
            and chunk["uri"].endswith(".py")
            for chunk in result
        )


@pytest.mark.benchmark
class TestScopeFilteringOverhead:
    """Verify scope filtering overhead meets performance requirements.

    These tests compare filtered queries against baseline to ensure overhead
    stays within acceptable limits (<5ms for typical queries).
    """

    def test_language_filter_overhead_acceptable(
        self, benchmark_catalog: DuckDBCatalog, benchmark
    ) -> None:
        """Verify language filter overhead is <5ms.

        Compares language-filtered query performance against baseline and
        asserts that overhead is within acceptable threshold.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        # Baseline: no filtering
        baseline_result = benchmark.pedantic(
            lambda: benchmark_catalog.query_by_ids(chunk_ids),
            iterations=100,
            rounds=10,
        )

        # Filtered: language filter
        filtered_result = benchmark.pedantic(
            lambda: benchmark_catalog.query_by_filters(chunk_ids, languages=["python"]),
            iterations=100,
            rounds=10,
        )

        # Calculate overhead (in milliseconds)
        baseline_time_ms = baseline_result.stats.mean * 1000
        filtered_time_ms = filtered_result.stats.mean * 1000
        overhead_ms = filtered_time_ms - baseline_time_ms

        # Assert overhead is acceptable
        assert overhead_ms < MAX_FILTER_OVERHEAD_MS, (
            f"Language filter overhead {overhead_ms:.2f}ms exceeds threshold {MAX_FILTER_OVERHEAD_MS}ms"
        )

    def test_path_glob_filter_overhead_acceptable(
        self, benchmark_catalog: DuckDBCatalog, benchmark
    ) -> None:
        """Verify path glob filter overhead is <5ms.

        Compares path-glob-filtered query performance against baseline and
        asserts that overhead is within acceptable threshold.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        # Baseline: no filtering
        baseline_result = benchmark.pedantic(
            lambda: benchmark_catalog.query_by_ids(chunk_ids),
            iterations=100,
            rounds=10,
        )

        # Filtered: path glob filter
        filtered_result = benchmark.pedantic(
            lambda: benchmark_catalog.query_by_filters(chunk_ids, include_globs=["src/**"]),
            iterations=100,
            rounds=10,
        )

        # Calculate overhead (in milliseconds)
        baseline_time_ms = baseline_result.stats.mean * 1000
        filtered_time_ms = filtered_result.stats.mean * 1000
        overhead_ms = filtered_time_ms - baseline_time_ms

        # Assert overhead is acceptable
        assert overhead_ms < MAX_FILTER_OVERHEAD_MS, (
            f"Path glob filter overhead {overhead_ms:.2f}ms exceeds threshold {MAX_FILTER_OVERHEAD_MS}ms"
        )

    def test_combined_filter_overhead_acceptable(
        self, benchmark_catalog: DuckDBCatalog, benchmark
    ) -> None:
        """Verify combined filter overhead is <5ms.

        Compares combined (language + path) filter query performance against
        baseline and asserts that overhead is within acceptable threshold.

        Parameters
        ----------
        benchmark_catalog : DuckDBCatalog
            Catalog with 100K chunks loaded.
        benchmark : pytest_benchmark.fixture.BenchmarkFixture
            pytest-benchmark fixture for timing.
        """
        chunk_ids = list(range(1, QUERY_CHUNK_COUNT + 1))

        # Baseline: no filtering
        baseline_result = benchmark.pedantic(
            lambda: benchmark_catalog.query_by_ids(chunk_ids),
            iterations=100,
            rounds=10,
        )

        # Filtered: combined filters
        filtered_result = benchmark.pedantic(
            lambda: benchmark_catalog.query_by_filters(
                chunk_ids,
                include_globs=["src/**"],
                languages=["python"],
            ),
            iterations=100,
            rounds=10,
        )

        # Calculate overhead (in milliseconds)
        baseline_time_ms = baseline_result.stats.mean * 1000
        filtered_time_ms = filtered_result.stats.mean * 1000
        overhead_ms = filtered_time_ms - baseline_time_ms

        # Assert overhead is acceptable
        assert overhead_ms < MAX_FILTER_OVERHEAD_MS, (
            f"Combined filter overhead {overhead_ms:.2f}ms exceeds threshold {MAX_FILTER_OVERHEAD_MS}ms"
        )
