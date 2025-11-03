"""Typed helpers for working with the AutoAPI parser module."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Protocol, cast

__all__ = ["AutoapiParserProtocol", "coerce_parser_class"]

_MISSING = object()


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
    candidate_obj = getattr(module, attribute, _MISSING)
    if candidate_obj is _MISSING or not inspect.isclass(candidate_obj):
        message = f"Module '{module.__name__}' attribute '{attribute}' is not a class"
        raise TypeError(message)
    return cast(type[AutoapiParserProtocol], candidate_obj)
