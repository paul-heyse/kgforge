"""Internal typed models for documentation artifacts.

This private package exports authoritative models for documentation artifact
pipelines. Models are built on Pydantic V2 BaseModel for full type safety,
self-documenting schemas, and built-in validation.

Sub-modules:

- **artifacts**: Authoritative Pydantic models for symbol index, delta, and
  reverse-lookup artifacts. These models are frozen (immutable) and include
  field validators for defensive validation.
- **griffe**: Runtime-checkable protocols and typed facade for Griffe
  integration (optional dependency).
- **sphinx_optional**: Typed facades for optional Sphinx/Astroid dependencies
  with graceful handling of missing imports.
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
from docs._types.astroid_facade import (
    AstroidBuilderProtocol,
    AstroidManagerProtocol,
    coerce_astroid_builder_class,
    coerce_astroid_manager_class,
)
from docs._types.griffe import (
    GriffeFacade,
    GriffeNode,
    LoaderFacade,
    MemberIterator,
    build_facade,
)
from docs._types.sphinx_optional import (
    AstroidManagerFacade,
    AutoapiParserFacade,
    MissingDependencyError,
    OptionalDependencies,
    load_optional_dependencies,
)

__all__ = [
    "ArtifactValidationError",
    "AstroidBuilderProtocol",
    "AstroidManagerFacade",
    "AstroidManagerProtocol",
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
    "coerce_astroid_builder_class",
    "coerce_astroid_manager_class",
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
