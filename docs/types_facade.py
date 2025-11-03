# ruff: noqa: PLC2701

"""Public facade re-exporting typed helpers for documentation tooling.

This module wraps the internal ``docs._types`` package so that other modules
can import the typed helpers without referencing private module names.  It is a
thin layer that re-exports the symbols required by ``docs/conf.py`` and other
documentation tooling.
"""

from __future__ import annotations

from docs._types.astroid_facade import (
    AstroidBuilderProtocol,
    AstroidManagerProtocol,
    coerce_astroid_builder_class,
    coerce_astroid_manager_class,
)
from docs._types.autoapi_parser import (
    AutoapiParserProtocol,
    coerce_parser_class,
)

__all__ = [
    "AstroidBuilderProtocol",
    "AstroidManagerProtocol",
    "AutoapiParserProtocol",
    "coerce_astroid_builder_class",
    "coerce_astroid_manager_class",
    "coerce_parser_class",
]
