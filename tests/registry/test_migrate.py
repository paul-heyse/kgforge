from __future__ import annotations

from pathlib import Path

import pytest

from kgfoundry_common.errors import RegistryError
from registry import duckdb_helpers, migrate


def test_apply_runs_all_migrations(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_init.sql").write_text(
        """
        CREATE TABLE alpha(id INT);
        INSERT INTO alpha VALUES (1);
        """
    )

    db_path = tmp_path / "registry.duckdb"
    migrate.apply(str(db_path), str(migrations_dir))

    connection = duckdb_helpers.connect(db_path, read_only=True)
    try:
        rows = duckdb_helpers.fetch_all(
            connection,
            "SELECT id FROM alpha",
            operation="tests.registry.migrate_fetch",
            require_parameterized=False,
        )
    finally:
        connection.close()

    assert rows == [(1,)]


def test_apply_raises_registry_error_on_failure(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_bad.sql").write_text("INSERT INTO missing_table VALUES (1);")

    db_path = tmp_path / "registry.duckdb"

    with pytest.raises(RegistryError):
        migrate.apply(str(db_path), str(migrations_dir))
