"""Module for registry.migrate.

NavMap:
- NavMap: Structure describing a module navmap.
- apply: Return apply.
- main: Run the CLI entry point for migration commands.
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
    "synopsis": "Module for registry.migrate",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["apply", "main"],
        },
    ],
}


# [nav:anchor apply]
def apply(db: str, migrations_dir: str) -> None:
    """Return apply.

    Parameters
    ----------
    db : str
        Description.
    migrations_dir : str
        Description.
    """
    con = duckdb.connect(db)
    for p in sorted(pathlib.Path(migrations_dir).glob("*.sql")):
        con.execute(p.read_text())
    con.close()


# [nav:anchor main]
def main() -> None:
    """Run the CLI entry point for migration commands."""
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
