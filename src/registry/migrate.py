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
    """Compute apply.

    Carry out the apply operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    db : str
        Description for ``db``.
    migrations_dir : str
        Description for ``migrations_dir``.
    
    Raises
    ------
    Exception
        Raised when validation fails.
    
    Examples
    --------
    >>> from registry.migrate import apply
    >>> apply(..., ...)  # doctest: +ELLIPSIS
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
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Examples
    --------
    >>> from registry.migrate import main
    >>> main()  # doctest: +ELLIPSIS
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
