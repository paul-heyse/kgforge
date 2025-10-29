"""Module entrypoint for ``python -m tools.docstring_builder``."""

from __future__ import annotations

from tools.docstring_builder.cli import main


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    run()
