"""Typed payload models for documentation tooling artifacts."""

from __future__ import annotations

from typing import Final

import msgspec

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


class SourceLinkModel(msgspec.Struct, frozen=True, omit_defaults=True):
    """Schema model describing source link bundles attached to symbols."""

    editor: str | None = None
    github: str | None = None


class SymbolIndexRowModel(msgspec.Struct, frozen=True, omit_defaults=True):
    """Schema model describing a single symbol index row."""

    path: str
    canonical_path: str | None = None
    kind: str
    package: str | None = None
    module: str | None = None
    file: str | None = None
    lineno: int | None = None
    endlineno: int | None = None
    doc: str
    signature: str | None = None
    is_async: bool = False
    is_property: bool = False
    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    section: str | None = None
    tested_by: list[str] = msgspec.field(default_factory=list)
    source_link: dict[str, str] = msgspec.field(default_factory=dict)


class SymbolDeltaChangeModel(msgspec.Struct, frozen=True, omit_defaults=True):
    """Schema model describing a change entry within a delta payload."""

    path: str
    before: dict[str, JsonValue] = msgspec.field(default_factory=dict)
    after: dict[str, JsonValue] = msgspec.field(default_factory=dict)
    reasons: list[str] = msgspec.field(default_factory=list)


class SymbolDeltaPayloadModel(msgspec.Struct, frozen=True, omit_defaults=True):
    """Schema model describing the symbol delta payload."""

    base_sha: str | None = None
    head_sha: str | None = None
    added: list[str] = msgspec.field(default_factory=list)
    removed: list[str] = msgspec.field(default_factory=list)
    changed: list[SymbolDeltaChangeModel] = msgspec.field(default_factory=list)


__all__: Final[list[str]] = [
    "JsonPrimitive",
    "JsonValue",
    "SourceLinkModel",
    "SymbolDeltaChangeModel",
    "SymbolDeltaPayloadModel",
    "SymbolIndexRowModel",
]
