"""Typed helpers for working with the AutoAPI parser module."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Protocol, cast

__all__ = ["AutoapiParserProtocol", "coerce_parser_class"]


class AutoapiParserProtocol(Protocol):
    """Subset of the AutoAPI parser surface used in the docs build."""

    def parse(self, node: object, /) -> object:
        """Return an AutoAPI document tree for ``node``."""

    def _parse_file(self, file_path: str, condition: Callable[[str], bool], /) -> object:
        """Parse ``file_path`` when ``condition`` evaluates to ``True``."""

    def parse_file(self, file_path: str, /) -> object:
        """Wrap :meth:`_parse_file` for convenience."""

    def parse_file_in_namespace(self, file_path: str, dir_root: str, /) -> object:
        """Parse ``file_path`` relative to ``dir_root`` namespace."""


def coerce_parser_class(
    module: ModuleType,
    attribute: str = "Parser",
) -> type[AutoapiParserProtocol]:
    """Return the ``Parser`` class from ``module`` with precise typing."""
    candidate_any = getattr(module, attribute, None)  # type: ignore[misc]
    if candidate_any is None or not isinstance(candidate_any, type):
        message = f"Module '{module.__name__}' attribute '{attribute}' is not a class"
        raise TypeError(message)
    parser_type = cast(type[object], candidate_any)
    return cast(type[AutoapiParserProtocol], parser_type)  # type: ignore[misc]
