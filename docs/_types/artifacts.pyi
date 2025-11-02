"""Stub file for typed artifact models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

type JsonPrimitive = str | int | float | bool | None
if TYPE_CHECKING:
    type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
else:
    JsonValue = object

type JsonPayload = Mapping[str, JsonValue] | Sequence[JsonValue] | JsonValue

@dataclass(frozen=True, slots=True)
class LineSpan:
    """Start/end line numbers for a symbol."""

    start: int | None
    end: int | None

class SymbolIndexRow:
    """A single symbol entry in the index."""

    path: str
    kind: str
    doc: str
    tested_by: tuple[str, ...]
    source_link: dict[str, str]
    canonical_path: str | None
    module: str | None
    package: str | None
    file: str | None
    span: LineSpan | None
    signature: str | None
    owner: str | None
    stability: str | None
    since: str | None
    deprecated_in: str | None
    section: str | None
    is_async: bool
    is_property: bool

    def __init__(
        self,
        *,
        path: str,
        kind: str,
        doc: str,
        tested_by: tuple[str, ...],
        source_link: dict[str, str],
        canonical_path: str | None = None,
        module: str | None = None,
        package: str | None = None,
        file: str | None = None,
        span: LineSpan | None = None,
        signature: str | None = None,
        owner: str | None = None,
        stability: str | None = None,
        since: str | None = None,
        deprecated_in: str | None = None,
        section: str | None = None,
        is_async: bool = ...,
        is_property: bool = ...,
    ) -> None: ...

class SymbolIndexArtifacts:
    """Complete symbol index payload with forward and reverse lookups."""

    rows: tuple[SymbolIndexRow, ...]
    by_file: dict[str, tuple[str, ...]]
    by_module: dict[str, tuple[str, ...]]

    def __init__(
        self,
        *,
        rows: tuple[SymbolIndexRow, ...],
        by_file: dict[str, tuple[str, ...]],
        by_module: dict[str, tuple[str, ...]],
    ) -> None: ...

class SymbolDeltaChange:
    """A single changed symbol between two versions."""

    path: str
    before: dict[str, JsonValue]
    after: dict[str, JsonValue]
    reasons: tuple[str, ...]

    def __init__(
        self,
        *,
        path: str,
        before: dict[str, JsonValue],
        after: dict[str, JsonValue],
        reasons: tuple[str, ...],
    ) -> None: ...

class SymbolDeltaPayload:
    """Delta (diff) of symbols between two git commits or documentation builds."""

    base_sha: str | None
    head_sha: str | None
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[SymbolDeltaChange, ...]

    def __init__(
        self,
        *,
        base_sha: str | None,
        head_sha: str | None,
        added: tuple[str, ...],
        removed: tuple[str, ...],
        changed: tuple[SymbolDeltaChange, ...],
    ) -> None: ...

class ArtifactValidationError(RuntimeError):
    """Raised when an artifact fails schema validation or conversion."""

    artifact_name: str | None
    problem_details: dict[str, JsonValue] | None

    def __init__(
        self,
        message: str,
        artifact_name: str | None = None,
        problem_details: dict[str, JsonValue] | None = None,
    ) -> None: ...

def symbol_index_from_json(raw: JsonPayload) -> SymbolIndexArtifacts: ...
def symbol_index_to_payload(
    model: SymbolIndexArtifacts,
) -> list[dict[str, JsonValue]]: ...
def symbol_delta_from_json(raw: JsonPayload) -> SymbolDeltaPayload: ...
def symbol_delta_to_payload(model: SymbolDeltaPayload) -> dict[str, JsonValue]: ...
def load_symbol_index(path: Path) -> SymbolIndexArtifacts: ...
def dump_symbol_index(path: Path, model: SymbolIndexArtifacts) -> None: ...
def load_symbol_delta(path: Path) -> SymbolDeltaPayload: ...
def dump_symbol_delta(path: Path, model: SymbolDeltaPayload) -> None: ...
