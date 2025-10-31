"""Overview of migrate.

This module bundles migrate logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import argparse
import pathlib
from contextlib import closing
from typing import Final, cast

from kgfoundry_common.errors import RegistryError
from kgfoundry_common.navmap_types import NavMap
from registry import duckdb_helpers

__all__ = ["apply", "main"]

__navmap__: Final[NavMap] = {
    "title": "registry.migrate",
    "synopsis": "Migration helpers for DuckDB registry schemas",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@registry",
        "stability": "experimental",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@registry",
            "stability": "experimental",
            "since": "0.1.0",
        }
        for name in __all__
    },
}


# [nav:anchor apply]
def apply(db: str, migrations_dir: str) -> None:
    """Describe apply.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    db : str
        Describe ``db``.
    migrations_dir : str
        Describe ``migrations_dir``.
    """
    path = pathlib.Path(migrations_dir)
    if not path.exists():
        error_message = "Migrations directory does not exist"
        raise RegistryError(
            error_message,
            context={"migrations_dir": str(path.resolve())},
        )

    with closing(duckdb_helpers.connect(db, pragmas={"threads": 4})) as con:
        for migration in sorted(path.glob("*.sql")):
            sql = migration.read_text()
            statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
            for statement in statements:
                try:
                    duckdb_helpers.execute(
                        con,
                        statement,
                        params=None,
                        require_parameterized=False,
                        operation=f"registry.migrate.{migration.stem}",
                    )
                except RegistryError as err:
                    message = err.message.lower()
                    if "read_parquet" in message and "table function" in message:
                        continue
                    raise


# [nav:anchor main]
def main() -> None:
    """Describe main.

    <!-- auto:docstring-builder v1 -->

    Python's object protocol for this class. Use it to integrate with built-in operators, protocols,
    or runtime behaviours that expect instances to participate in the language's data model.
    """
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("apply")
    a.add_argument("--db", required=True)
    a.add_argument("--migrations", required=True)
    ns = ap.parse_args()
    db_arg = cast(str, ns.db)
    migrations_arg = cast(str, ns.migrations)
    cmd = cast(str, ns.cmd)
    if cmd == "apply":
        apply(db_arg, migrations_arg)


if __name__ == "__main__":
    main()
