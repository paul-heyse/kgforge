"""Typed re-exports for :mod:`docs._types.artifacts`."""

from __future__ import annotations

from docs._types.alignment import (
    SYMBOL_DELTA_CHANGE_FIELDS,
    SYMBOL_DELTA_PAYLOAD_FIELDS,
    SYMBOL_INDEX_ARTIFACTS_FIELDS,
    SYMBOL_INDEX_ROW_FIELDS,
    align_schema_fields,
)
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

__all__ = [
    "SYMBOL_DELTA_CHANGE_FIELDS",
    "SYMBOL_DELTA_PAYLOAD_FIELDS",
    "SYMBOL_INDEX_ARTIFACTS_FIELDS",
    "SYMBOL_INDEX_ROW_FIELDS",
    "ArtifactValidationError",
    "JsonPayload",
    "JsonPrimitive",
    "JsonValue",
    "LineSpan",
    "SymbolDeltaChange",
    "SymbolDeltaPayload",
    "SymbolIndexArtifacts",
    "SymbolIndexRow",
    "align_schema_fields",
    "dump_symbol_delta",
    "dump_symbol_index",
    "load_symbol_delta",
    "load_symbol_index",
    "symbol_delta_from_json",
    "symbol_delta_to_payload",
    "symbol_index_from_json",
    "symbol_index_to_payload",
]
