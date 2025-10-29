"""Overview of cli.

This module bundles cli logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final

import typer

from kgfoundry_common.navmap_types import NavMap

__all__ = ["harvest"]

__navmap__: Final[NavMap] = {
    "title": "download.cli",
    "synopsis": "Command-line entrypoints for bulk download orchestration",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@download",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        "harvest": {
            "owner": "@download",
            "stability": "experimental",
            "since": "0.2.0",
        },
    },
}

app = typer.Typer(help="Downloader & harvester CLI (skeleton).")


# [nav:anchor harvest]
def harvest(topic: str, years: str = ">=2018", max_works: int = 20000) -> None:
    """Compute harvest.
<!-- auto:docstring-builder v1 -->

Carry out the harvest operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

Parameters
----------
topic : str
    Description for ``topic``.
years : str, optional
    Defaults to ``'>=2018'``.
    Description for ``years``.
    Defaults to ``'>=2018'``.
max_works : int, optional
    Defaults to ``20000``.
    Description for ``max_works``.
    
    
    
    Defaults to ``20000``.

Examples
--------
>>> from download.cli import harvest
>>> harvest(...)  # doctest: +ELLIPSIS
"""
    typer.echo(f"[dry-run] would harvest topic={topic!r}, years={years}, max_works={max_works}")


app.command()(harvest)


if __name__ == "__main__":
    app()
