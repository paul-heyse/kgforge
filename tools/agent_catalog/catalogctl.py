"""Compatibility wrapper delegating to :mod:`kgfoundry.agent_catalog.cli`."""

from __future__ import annotations

from kgfoundry.agent_catalog.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
