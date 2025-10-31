"""Compatibility wrapper delegating to :mod:`kgfoundry.agent_catalog.mcp`."""

from __future__ import annotations

from kgfoundry.agent_catalog.mcp import build_parser, main

__all__ = ["build_parser", "main"]

JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
JsonObject = dict[str, JsonValue]

if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
