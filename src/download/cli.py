"""Overview of cli.

This module bundles cli logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import typer

if TYPE_CHECKING:
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
    """Harvest documents from OpenAlex matching the given criteria.

    Downloads research papers from OpenAlex filtered by topic and
    publication year. Currently implements a dry-run mode.

    Parameters
    ----------
    topic : str
        Topic query string for filtering papers.
    years : str, optional
        Year filter expression (e.g., ">=2018"). Defaults to ">=2018".
    max_works : int, optional
        Maximum number of works to harvest. Defaults to 20000.
    """
    typer.echo(f"[dry-run] would harvest topic={topic!r}, years={years}, max_works={max_works}")


app.command()(harvest)


if __name__ == "__main__":
    app()
