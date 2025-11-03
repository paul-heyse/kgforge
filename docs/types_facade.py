"""Public facade re-exporting typed helpers for documentation tooling.

This module wraps the internal ``docs._types`` package so that other modules
can import the typed helpers without referencing private module names.  It is a
thin layer that re-exports the symbols required by ``docs/conf.py`` and other
documentation tooling.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

_ASTROID_FACADE = import_module("docs._types.astroid_facade")
_AUTOAPI_PARSER = import_module("docs._types.autoapi_parser")

AstroidBuilderProtocol = cast(Any, _ASTROID_FACADE.AstroidBuilderProtocol)
AstroidManagerProtocol = cast(Any, _ASTROID_FACADE.AstroidManagerProtocol)
coerce_astroid_builder_class = cast(Any, _ASTROID_FACADE.coerce_astroid_builder_class)
coerce_astroid_manager_class = cast(Any, _ASTROID_FACADE.coerce_astroid_manager_class)
AutoapiParserProtocol = cast(Any, _AUTOAPI_PARSER.AutoapiParserProtocol)
coerce_parser_class = cast(Any, _AUTOAPI_PARSER.coerce_parser_class)

__all__ = [
    "AstroidBuilderProtocol",
    "AstroidManagerProtocol",
    "AutoapiParserProtocol",
    "coerce_astroid_builder_class",
    "coerce_astroid_manager_class",
    "coerce_parser_class",
]
