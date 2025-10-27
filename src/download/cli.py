"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
download.cli
"""


from __future__ import annotations

from typing import Final

import typer

from kgfoundry_common.navmap_types import NavMap

__all__ = ["harvest"]

__navmap__: Final[NavMap] = {
    "title": "download.cli",
    "synopsis": "Module for download.cli",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["harvest"],
        },
    ],
}

app = typer.Typer(help="Downloader & harvester CLI (skeleton).")


# [nav:anchor harvest]
@app.command()
def harvest(topic: str, years: str = ">=2018", max_works: int = 20000) -> None:
    """
    Return harvest.
    
    Parameters
    ----------
    topic : str
        Description for ``topic``.
    years : str, optional
        Description for ``years``.
    max_works : int, optional
        Description for ``max_works``.
    
    Examples
    --------
    >>> from download.cli import harvest
    >>> harvest(..., ..., ...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    download.cli
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    typer.echo(f"[dry-run] would harvest topic={topic!r}, years={years}, max_works={max_works}")


if __name__ == "__main__":
    app()
