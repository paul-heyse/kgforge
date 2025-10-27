"""Module for download.cli.

NavMap:
- harvest: Harvest OA metadata and download PDFs (skeleton).
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Downloader & harvester CLI (skeleton).")


@app.command()
def harvest(topic: str, years: str = ">=2018", max_works: int = 20000) -> None:
    """Harvest OA metadata and download PDFs (skeleton)."""
    typer.echo(f"[dry-run] would harvest topic={topic!r}, years={years}, max_works={max_works}")


if __name__ == "__main__":
    app()
