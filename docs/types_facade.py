"""Public facade re-exporting typed helpers for documentation tooling.

This module wraps the internal ``docs._types`` package so that other modules
can import the typed helpers without referencing private module names.  It is a
thin layer that re-exports the symbols required by ``docs/conf.py`` and other
documentation tooling.
"""

from __future__ import annotations

from docs._types.astroid_facade import (
    AstroidBuilderFactory,
    AstroidBuilderProtocol,
    AstroidManagerFactory,
    AstroidManagerProtocol,
    coerce_astroid_builder_factory,
    coerce_astroid_manager_factory,
)
from docs._types.autoapi_parser import (
    AutoapiParserProtocol,
    coerce_parser_class,
)

__all__ = [
    "AstroidBuilderFactory",
    "AstroidBuilderProtocol",
    "AstroidManagerFactory",
    "AstroidManagerProtocol",
    "AutoapiParserProtocol",
    "coerce_astroid_builder_factory",
    "coerce_astroid_manager_factory",
    "coerce_parser_class",
]
