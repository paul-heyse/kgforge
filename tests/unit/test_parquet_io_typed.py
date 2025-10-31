"""Unit tests for typed Parquet IO helpers.

Tests verify typed table extraction, DataFrame conversion, and schema validation.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kgfoundry_common.errors import DeserializationError
from kgfoundry_common.parquet_io import (
    read_table,
    read_table_to_dataframe,
    validate_table_schema,
)


@pytest.fixture
def sample_schema() -> pa.Schema:
    """Create a sample schema for testing."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("value", pa.int32()),
        ]
    )


@pytest.fixture
def sample_table(sample_schema: pa.Schema) -> pa.Table:
    """Create a sample table for testing."""
    return pa.Table.from_pylist(
        [{"id": "test1", "value": 42}, {"id": "test2", "value": 100}],
        schema=sample_schema,
    )


@pytest.fixture
def parquet_file(tmp_path: Path, sample_table: pa.Table) -> Path:
    """Create a temporary Parquet file."""
    file_path = tmp_path / "test.parquet"
    pq.write_table(sample_table, file_path)
    return file_path


def test_read_table_success(parquet_file: Path, sample_schema: pa.Schema) -> None:
    """Test reading a Parquet file returns typed Table.

    Scenario: Typed table extraction
    - Maps to Requirement: Parquet IO Type Safety (R6)
    """
    table = read_table(parquet_file)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert table.num_columns == 2
    assert table.schema.equals(sample_schema)


def test_read_table_with_schema_validation(parquet_file: Path, sample_schema: pa.Schema) -> None:
    """Test reading with schema validation succeeds for matching schema."""
    table = read_table(parquet_file, schema=sample_schema, validate_schema=True)
    assert table.schema.equals(sample_schema)


def test_read_table_schema_mismatch(parquet_file: Path) -> None:
    """Test reading with schema validation raises error on mismatch."""
    wrong_schema = pa.schema([pa.field("wrong", pa.string())])
    with pytest.raises(DeserializationError, match="Schema mismatch"):
        read_table(parquet_file, schema=wrong_schema, validate_schema=True)


def test_read_table_file_not_found() -> None:
    """Test reading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_table("nonexistent.parquet")


def test_read_table_to_dataframe_success(parquet_file: Path, sample_schema: pa.Schema) -> None:
    """Test reading to DataFrame returns typed DataFrame.

    Scenario: DataFrame conversion retains typing
    - Maps to Requirement: Parquet IO Type Safety (R6)
    """
    pytest.importorskip("pandas")
    df = read_table_to_dataframe(parquet_file)
    assert len(df) == 2
    expected_columns = [field.name for field in sample_schema]
    assert list(df.columns) == expected_columns
    assert df["id"].dtype == "object"
    assert df["value"].dtype == "int32"


def test_read_table_to_dataframe_with_schema(parquet_file: Path, sample_schema: pa.Schema) -> None:
    """Test reading to DataFrame with schema validation."""
    pytest.importorskip("pandas")
    df = read_table_to_dataframe(parquet_file, schema=sample_schema, validate_schema=True)
    assert len(df) == 2


def test_read_table_to_dataframe_no_pandas(parquet_file: Path) -> None:
    """Test reading to DataFrame without pandas raises ImportError."""
    # This test verifies error handling, but pandas is usually available
    # In practice, this would only fail if pandas is not installed
    try:
        import pandas  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pandas is required"):
            read_table_to_dataframe(parquet_file)


def test_validate_table_schema_success(sample_table: pa.Table, sample_schema: pa.Schema) -> None:
    """Test schema validation succeeds for matching schema."""
    validate_table_schema(sample_table, sample_schema)  # No error


def test_validate_table_schema_mismatch(sample_table: pa.Table) -> None:
    """Test schema validation raises error on mismatch."""
    wrong_schema = pa.schema([pa.field("wrong", pa.string())])
    with pytest.raises(DeserializationError, match="Schema mismatch"):
        validate_table_schema(sample_table, wrong_schema)


def test_read_table_skip_validation(parquet_file: Path) -> None:
    """Test reading without schema validation skips checks."""
    wrong_schema = pa.schema([pa.field("wrong", pa.string())])
    # Should not raise error when validate_schema=False
    table = read_table(parquet_file, schema=wrong_schema, validate_schema=False)
    assert isinstance(table, pa.Table)


def test_read_table_pathlib_path(parquet_file: Path, sample_schema: pa.Schema) -> None:
    """Test reading accepts Path objects."""
    table = read_table(parquet_file, schema=sample_schema)
    assert isinstance(table, pa.Table)


def test_read_table_str_path(tmp_path: Path, sample_table: pa.Table) -> None:
    """Test reading accepts string paths."""
    file_path = tmp_path / "test_str.parquet"
    pq.write_table(sample_table, file_path)
    table = read_table(str(file_path))
    assert isinstance(table, pa.Table)
