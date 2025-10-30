"""Overview of migrate.

This module bundles migrate logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Final

import duckdb

from kgfoundry_common.navmap_types import NavMap

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
    con = duckdb.connect(db)
    for p in sorted(pathlib.Path(migrations_dir).glob("*.sql")):
        sql = p.read_text()
        statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
        for statement in statements:
            try:
                con.execute(statement)
            except duckdb.Error as exc:
                message = str(exc).lower()
                if "read_parquet" in message and "table function" in message:
                    # DuckDB 1.1 disallows non-constant arguments to table functions.
                    # Later migrations may re-create these views once a compatible
                    # approach is available, so skip them for now while still
                    # applying schema changes.
                    continue
                raise
    con.close()


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
    if ns.cmd == "apply":
        apply(ns.db, ns.migrations)


if __name__ == "__main__":
    main()
