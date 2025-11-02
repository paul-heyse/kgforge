"""Typed models and facades for documentation artifact pipeline.

This package provides:

- **artifacts**: Authoritative msgspec models for symbol index, delta, and reverse-lookup artifacts.
- **griffe**: Runtime-checkable protocols and loader facades for Griffe integration.
- **sphinx_optional**: Typed facades for optional Sphinx dependencies (Astroid, AutoAPI).

Examples
--------
>>> from docs._types.artifacts import symbol_index_from_json, symbol_index_to_payload
>>> from docs._types.griffe import build_facade
>>> from docs._types.sphinx_optional import load_optional_dependencies
"""

from __future__ import annotations

from docs._types import artifacts as artifacts
from docs._types import griffe as griffe
from docs._types import sphinx_optional as sphinx_optional
from docs._types.artifacts import (
    ArtifactValidationError,
    JsonPayload,
    JsonPrimitive,
    JsonValue,
    LineSpan,
    SymbolDeltaChange,
    SymbolDeltaPayload,
    SymbolIndexArtifacts,
    SymbolIndexRow,
    dump_symbol_delta,
    dump_symbol_index,
    load_symbol_delta,
    load_symbol_index,
    symbol_delta_from_json,
    symbol_delta_to_payload,
    symbol_index_from_json,
    symbol_index_to_payload,
)
from docs._types.griffe import (
    AutoapiParserFacade,
    GriffeFacade,
    GriffeNode,
    LoaderFacade,
    MemberIterator,
    build_facade,
)
from docs._types.sphinx_optional import (
    AstroidManagerFacade,
    MissingDependencyError,
    OptionalDependencies,
    load_optional_dependencies,
)

__all__ = [
    "ArtifactValidationError",
    "AstroidManagerFacade",
    "AutoapiParserFacade",
    "GriffeFacade",
    "GriffeNode",
    "JsonPayload",
    "JsonPrimitive",
    "JsonValue",
    "LineSpan",
    "LoaderFacade",
    "MemberIterator",
    "MissingDependencyError",
    "OptionalDependencies",
    "SymbolDeltaChange",
    "SymbolDeltaPayload",
    "SymbolIndexArtifacts",
    "SymbolIndexRow",
    "build_facade",
    "dump_symbol_delta",
    "dump_symbol_index",
    "load_optional_dependencies",
    "load_symbol_delta",
    "load_symbol_index",
    "symbol_delta_from_json",
    "symbol_delta_to_payload",
    "symbol_index_from_json",
    "symbol_index_to_payload",
]
