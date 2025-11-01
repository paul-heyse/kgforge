"""Run import-linter against the generated tooling contracts."""

from __future__ import annotations

from pathlib import Path

from tools.make_importlinter import main as generate_importlinter


def run() -> Path:
    """Generate the configuration and run import-linter in check mode."""
    return generate_importlinter(check=True)


if __name__ == "__main__":  # pragma: no cover
    run()
