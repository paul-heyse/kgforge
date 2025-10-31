from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from duckdb import DuckDBPyConnection

from kgfoundry_common.errors import RegistryError
from registry import duckdb_helpers


def _connect(db_path: Path) -> DuckDBPyConnection:
    return duckdb_helpers.connect(db_path)


def test_execute_with_parameters_inserts_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "helpers.duckdb"
    connection = _connect(db_path)
    try:
        duckdb_helpers.execute(
            connection,
            "CREATE TABLE IF NOT EXISTS items(id INT, name TEXT)",
            operation="tests.registry.create_table",
            require_parameterized=False,
        )

        duckdb_helpers.execute(
            connection,
            "INSERT INTO items VALUES (?, ?)",
            [1, "alpha"],
            operation="tests.registry.insert_item",
        )

        raw_results = cast(
            list[tuple[int, str]],
            duckdb_helpers.fetch_all(
                connection,
                "SELECT id, name FROM items ORDER BY id",
                operation="tests.registry.fetch_items",
            ),
        )
    finally:
        connection.close()

    assert raw_results == [(1, "alpha")]


def test_execute_requires_parameterized_queries(tmp_path: Path) -> None:
    db_path = tmp_path / "helpers.duckdb"
    connection = _connect(db_path)
    try:
        duckdb_helpers.execute(
            connection,
            "CREATE TABLE requires_param(id INT)",
            operation="tests.registry.create_requires_param",
            require_parameterized=False,
        )

        with pytest.raises(RegistryError) as exc:
            duckdb_helpers.execute(
                connection,
                "INSERT INTO requires_param VALUES (1)",
                [1],
                operation="tests.registry.missing_placeholder",
            )
    finally:
        connection.close()

    assert "parameterized" in str(exc.value).lower()


def test_execute_wraps_duckdb_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "helpers.duckdb"
    connection = _connect(db_path)
    try:
        with pytest.raises(RegistryError) as exc:
            duckdb_helpers.execute(
                connection,
                "SELECT * FROM missing_table",
                operation="tests.registry.missing_table",
                require_parameterized=False,
            )
    finally:
        connection.close()

    assert "missing_table" in str(exc.value)


def test_fetch_one_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "helpers.duckdb"
    connection = _connect(db_path)
    try:
        duckdb_helpers.execute(
            connection,
            "CREATE TABLE singleton(id INT)",
            operation="tests.registry.create_singleton",
            require_parameterized=False,
        )

        raw_row = cast(
            tuple[int] | None,
            duckdb_helpers.fetch_one(
                connection,
                "SELECT id FROM singleton",
                operation="tests.registry.fetch_singleton",
            ),
        )
    finally:
        connection.close()

    assert raw_row is None


def test_validate_identifier_allows_known_value() -> None:
    allowed = {"datasets", "runs"}
    assert duckdb_helpers.validate_identifier("runs", allowed, label="table") == "runs"

    with pytest.raises(RegistryError):
        duckdb_helpers.validate_identifier("documents", allowed, label="table")


def test_connect_applies_pragmas(tmp_path: Path) -> None:
    db_path = tmp_path / "pragmas.duckdb"
    with closing(duckdb_helpers.connect(db_path, pragmas={"threads": 2})) as connection:
        raw = cast(
            tuple[str, int] | None,
            duckdb_helpers.fetch_one(
                connection,
                "PRAGMA threads",
                operation="tests.registry.pragma_threads",
                require_parameterized=False,
            ),
        )
    assert raw is not None
    threads = raw
    assert threads[0] == "threads"
    assert threads[1] == 2
