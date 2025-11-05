"""Module entrypoint for ``python -m tools.docstring_builder``."""

from __future__ import annotations

from tools.docstring_builder.cli import main


def run() -> None:
    """Invoke the CLI and exit with its status code.

    Raises
    ------
    SystemExit
        Exits with the status code returned by the CLI main function.
    """
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    run()
