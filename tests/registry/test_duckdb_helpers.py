from contextlib import closing
from pathlib import Path

import pytest

from kgfoundry_common.errors import RegistryError
from registry import duckdb_helpers


@pytest.fixture
def conn(tmp_path: Path):
    db_path = tmp_path / "helpers.duckdb"
    connection = duckdb_helpers.connect(db_path)
    try:
        yield connection
    finally:
        connection.close()


def test_execute_with_parameters_inserts_rows(conn):
    duckdb_helpers.execute(
        conn,
        "CREATE TABLE IF NOT EXISTS items(id INT, name TEXT)",
        operation="tests.registry.create_table",
        require_parameterized=False,
    )

    duckdb_helpers.execute(
        conn,
        "INSERT INTO items VALUES (?, ?)",
        [1, "alpha"],
        operation="tests.registry.insert_item",
    )

    results = duckdb_helpers.fetch_all(
        conn,
        "SELECT id, name FROM items ORDER BY id",
        operation="tests.registry.fetch_items",
    )

    assert results == [(1, "alpha")]


def test_execute_requires_parameterized_queries(conn):
    duckdb_helpers.execute(
        conn,
        "CREATE TABLE requires_param(id INT)",
        operation="tests.registry.create_requires_param",
        require_parameterized=False,
    )

    with pytest.raises(RegistryError) as exc:
        duckdb_helpers.execute(
            conn,
            "INSERT INTO requires_param VALUES (1)",
            [1],
            operation="tests.registry.missing_placeholder",
        )
    assert "parameterized" in str(exc.value).lower()


def test_execute_wraps_duckdb_errors(conn):
    with pytest.raises(RegistryError) as exc:
        duckdb_helpers.execute(
            conn,
            "SELECT * FROM missing_table",
            operation="tests.registry.missing_table",
            require_parameterized=False,
        )
    assert "missing_table" in str(exc.value)


def test_fetch_one_returns_none(conn):
    duckdb_helpers.execute(
        conn,
        "CREATE TABLE singleton(id INT)",
        operation="tests.registry.create_singleton",
        require_parameterized=False,
    )

    row = duckdb_helpers.fetch_one(
        conn,
        "SELECT id FROM singleton",
        operation="tests.registry.fetch_singleton",
    )
    assert row is None


def test_validate_identifier_allows_known_value():
    allowed = {"datasets", "runs"}
    assert duckdb_helpers.validate_identifier("runs", allowed, label="table") == "runs"

    with pytest.raises(RegistryError):
        duckdb_helpers.validate_identifier("documents", allowed, label="table")


def test_connect_applies_pragmas(tmp_path: Path):
    db_path = tmp_path / "pragmas.duckdb"
    with closing(duckdb_helpers.connect(db_path, pragmas={"threads": 2})) as connection:
        threads = duckdb_helpers.fetch_one(
            connection,
            "PRAGMA threads",
            operation="tests.registry.pragma_threads",
            require_parameterized=False,
        )
    # DuckDB returns single row with key/value columns
    assert threads is not None
    assert threads[0] == "threads"
    assert threads[1] == 2
