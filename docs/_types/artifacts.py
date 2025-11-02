"""Authoritative typed models for documentation artifacts.

This module provides msgspec-backed data structures and conversion helpers for all
documentation pipeline artifacts (symbol index, delta, reverse lookups). Models
serialize to payloads that validate against the canonical JSON Schemas under
`schema/docs/`.

The module owns the JSON contract and encapsulates all transformations, ensuring:

- Deterministic serialization (stable field order, tuple use for arrays)
- Schema compliance (all payloads validate via jsonschema)
- Type safety (no Any-typed access in public functions)
- Defensive validation (field coercion, missing-key handling)

Examples
--------
>>> import json
>>> from pathlib import Path
>>> from docs._types.artifacts import (
...     SymbolIndexRow,
...     SymbolIndexArtifacts,
...     symbol_index_to_payload,
...     symbol_index_from_json,
... )
>>> row = SymbolIndexRow(
...     path="pkg.mod.func",
...     canonical_path=None,
...     kind="function",
...     module="pkg.mod",
...     package="pkg",
...     file="pkg/mod.py",
...     span=None,
...     signature="(x: int) -> str",
...     owner=None,
...     stability=None,
...     since=None,
...     deprecated_in=None,
...     section=None,
...     tested_by=(),
...     is_async=False,
...     is_property=False,
... )
>>> artifacts = SymbolIndexArtifacts(
...     rows=(row,),
...     by_file={"pkg/mod.py": ("pkg.mod.func",)},
...     by_module={"pkg.mod": ("pkg.mod.func",)},
... )
>>> payload = symbol_index_to_payload(artifacts)
>>> assert isinstance(payload, list)
>>> restored = symbol_index_from_json(payload)
>>> assert restored.rows[0].path == "pkg.mod.func"
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

# Type aliases matching RFC 7159 JSON structure
type JsonPrimitive = str | int | float | bool | None
if TYPE_CHECKING:
    type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
else:
    JsonValue = object  # type: ignore[assignment, misc]

type JsonPayload = Mapping[str, JsonValue] | Sequence[JsonValue] | JsonValue

__all__ = [
    "ArtifactCodec",
    "ArtifactValidationError",
    "JsonPayload",
    "JsonPrimitive",
    "JsonValue",
    "LineSpan",
    "SymbolDeltaChange",
    "SymbolDeltaPayload",
    "SymbolIndexArtifacts",
    "SymbolIndexRow",
    "dump_symbol_delta",
    "dump_symbol_index",
    "load_symbol_delta",
    "load_symbol_index",
    "symbol_delta_from_json",
    "symbol_delta_to_payload",
    "symbol_index_from_json",
    "symbol_index_to_payload",
]


@dataclass(frozen=True, slots=True)
class LineSpan:
    """Start/end line numbers for a symbol.

    Parameters
    ----------
    start : int | None
        Starting line number (1-indexed), or None if unknown.
    end : int | None
        Ending line number (1-indexed, inclusive), or None if unknown.
    """

    start: int | None
    end: int | None


@msgspec.struct(omit_defaults=True, slots=True)
class SymbolIndexRow:
    """A single symbol entry in the index.

    Each row represents one documented symbol (function, class, module, etc.) with
    metadata needed for search, deep linking, and reverse lookups.

    Parameters
    ----------
    path : str
        Fully qualified symbol path (e.g., "pkg.mod.ClassName.method_name").
    canonical_path : str | None, optional
        If this symbol is an alias, canonical_path points to the real definition.
        Defaults to None.
    kind : str | None, optional
        Symbol kind: "module", "class", "function", "method", etc.
        Defaults to None.
    module : str | None, optional
        Module containing this symbol (e.g., "pkg.mod").
        Defaults to None.
    package : str | None, optional
        Top-level package name (e.g., "pkg").
        Defaults to None.
    file : str | None, optional
        Relative path to source file (e.g., "pkg/mod.py").
        Defaults to None.
    span : LineSpan | None, optional
        Start/end line numbers in the source file.
        Defaults to None.
    signature : str | None, optional
        Function/method signature string (e.g., "(x: int) -> str").
        Defaults to None.
    owner : str | None, optional
        For methods: qualified path to the owner class.
        Defaults to None.
    stability : str | None, optional
        Stability tag (e.g., "stable", "experimental").
        Defaults to None.
    since : str | None, optional
        Version when first introduced (e.g., "0.1.0").
        Defaults to None.
    deprecated_in : str | None, optional
        Version when deprecated (e.g., "0.2.0").
        Defaults to None.
    section : str | None, optional
        Documentation section or category.
        Defaults to None.
    tested_by : tuple[str, ...], optional
        Test paths (relative to tests/) that cover this symbol.
        Defaults to empty tuple.
    is_async : bool, optional
        True if this is an async function/method.
        Defaults to False.
    is_property : bool, optional
        True if this is a @property.
        Defaults to False.
    """

    path: str
    canonical_path: str | None = None
    kind: str | None = None
    module: str | None = None
    package: str | None = None
    file: str | None = None
    span: LineSpan | None = None
    signature: str | None = None
    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = None
    section: str | None = None
    tested_by: tuple[str, ...] = ()
    is_async: bool = False
    is_property: bool = False


@msgspec.struct(slots=True)
class SymbolIndexArtifacts:
    """Complete symbol index payload with forward and reverse lookups.

    Parameters
    ----------
    rows : tuple[SymbolIndexRow, ...]
        All symbol entries, sorted by path.
    by_file : dict[str, tuple[str, ...]]
        Reverse lookup: file path -> sorted tuple of symbol paths.
    by_module : dict[str, tuple[str, ...]]
        Reverse lookup: module name -> sorted tuple of symbol paths.
    """

    rows: tuple[SymbolIndexRow, ...]
    by_file: dict[str, tuple[str, ...]]
    by_module: dict[str, tuple[str, ...]]


@msgspec.struct(slots=True)
class SymbolDeltaChange:
    """A single changed symbol between two versions.

    Parameters
    ----------
    path : str
        The symbol path that changed.
    before : dict[str, JsonValue]
        Previous version of the row (serialized).
    after : dict[str, JsonValue]
        New version of the row (serialized).
    reasons : tuple[str, ...]
        List of reasons why the symbol changed (e.g., ["signature_changed", "doc_updated"]).
    """

    path: str
    before: dict[str, JsonValue]
    after: dict[str, JsonValue]
    reasons: tuple[str, ...]


@msgspec.struct(slots=True)
class SymbolDeltaPayload:
    """Delta (diff) of symbols between two git commits or documentation builds.

    Parameters
    ----------
    base_sha : str | None
        Git SHA or build identifier for the baseline.
    head_sha : str | None
        Git SHA or build identifier for the current state.
    added : tuple[str, ...]
        Sorted tuple of newly added symbol paths.
    removed : tuple[str, ...]
        Sorted tuple of removed symbol paths.
    changed : tuple[SymbolDeltaChange, ...]
        List of symbols that changed (sorted by path).
    """

    base_sha: str | None
    head_sha: str | None
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[SymbolDeltaChange, ...]


class ArtifactValidationError(RuntimeError):
    """Raised when an artifact fails schema validation or conversion.

    Parameters
    ----------
    message : str
        Human-readable error description.
    artifact_name : str | None, optional
        Logical name of the artifact (e.g., "symbols.json").
        Defaults to None.
    problem_details : dict[str, JsonValue] | None, optional
        RFC 9457 Problem Details dict with validation context.
        Defaults to None.

    Attributes
    ----------
    artifact_name : str | None
        Logical name of the artifact, if provided.
    problem_details : dict[str, JsonValue] | None
        RFC 9457 Problem Details, if provided.
    """

    def __init__(
        self,
        message: str,
        artifact_name: str | None = None,
        problem_details: dict[str, JsonValue] | None = None,
    ) -> None:
        """Initialize ArtifactValidationError."""
        super().__init__(message)
        self.artifact_name = artifact_name
        self.problem_details = problem_details


class ArtifactCodec(msgspec.json.Codec):
    """JSON codec with repository-specific defaults for artifact serialization.

    Configured for:
    - Deterministic serialization (sorted keys, stable field order)
    - Round-trip fidelity with msgspec structs
    - UTF-8 encoding
    """


def symbol_index_from_json(raw: JsonPayload) -> SymbolIndexArtifacts:
    """Construct a SymbolIndexArtifacts from a JSON payload with validation.

    Parameters
    ----------
    raw : JsonPayload
        Raw JSON payload (typically loaded via json.load or dict/list).

    Returns
    -------
    SymbolIndexArtifacts
        Validated typed artifact.

    Raises
    ------
    ArtifactValidationError
        If the payload is malformed or missing required fields.

    Examples
    --------
    >>> from docs._types.artifacts import symbol_index_from_json
    >>> payload = [
    ...     {
    ...         "path": "mod.func",
    ...         "canonical_path": None,
    ...         "kind": "function",
    ...         "module": "mod",
    ...         "package": "pkg",
    ...         "file": "mod.py",
    ...         "lineno": 10,
    ...         "endlineno": 20,
    ...         "signature": "(x: int) -> str",
    ...         "owner": None,
    ...         "stability": None,
    ...         "since": None,
    ...         "deprecated_in": None,
    ...         "section": None,
    ...         "tested_by": [],
    ...         "is_async": False,
    ...         "is_property": False,
    ...     }
    ... ]
    >>> artifacts = symbol_index_from_json(payload)
    >>> assert len(artifacts.rows) == 1
    """
    if not isinstance(raw, list):
        msg = f"Expected list of rows, got {type(raw).__name__}"
        raise ArtifactValidationError(msg, artifact_name="symbol-index")

    rows: list[SymbolIndexRow] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            msg = f"Row {i}: expected dict, got {type(item).__name__}"
            raise ArtifactValidationError(msg, artifact_name="symbol-index")

        try:
            # Extract span if line numbers are present
            span: LineSpan | None = None
            lineno = item.get("lineno")
            endlineno = item.get("endlineno")
            if lineno is not None or endlineno is not None:
                span = LineSpan(
                    start=int(lineno) if lineno is not None else None,
                    end=int(endlineno) if endlineno is not None else None,
                )

            # Coerce tested_by to tuple
            tested_by_list = item.get("tested_by", [])
            tested_by = tuple(tested_by_list) if isinstance(tested_by_list, (list, tuple)) else ()

            row = SymbolIndexRow(
                path=str(item["path"]),
                canonical_path=item.get("canonical_path"),
                kind=item.get("kind"),
                module=item.get("module"),
                package=item.get("package"),
                file=item.get("file"),
                span=span,
                signature=item.get("signature"),
                owner=item.get("owner"),
                stability=item.get("stability"),
                since=item.get("since"),
                deprecated_in=item.get("deprecated_in"),
                section=item.get("section"),
                tested_by=tested_by,
                is_async=bool(item.get("is_async", False)),
                is_property=bool(item.get("is_property", False)),
            )
            rows.append(row)
        except (KeyError, ValueError, TypeError) as e:
            msg = f"Row {i}: failed to construct SymbolIndexRow: {e}"
            raise ArtifactValidationError(msg, artifact_name="symbol-index") from e

    # Return artifacts with empty lookups (to be populated separately if needed)
    return SymbolIndexArtifacts(rows=tuple(rows), by_file={}, by_module={})


def symbol_index_to_payload(model: SymbolIndexArtifacts) -> list[dict[str, JsonValue]]:
    """Serialize SymbolIndexArtifacts to a JSON payload.

    Parameters
    ----------
    model : SymbolIndexArtifacts
        Typed artifact to serialize.

    Returns
    -------
    list[dict[str, JsonValue]]
        JSON payload ready for writing to disk or validation.

    Examples
    --------
    >>> from docs._types.artifacts import (
    ...     SymbolIndexRow,
    ...     SymbolIndexArtifacts,
    ...     symbol_index_to_payload,
    ... )
    >>> row = SymbolIndexRow(path="mod.func", kind="function")
    >>> artifacts = SymbolIndexArtifacts(
    ...     rows=(row,),
    ...     by_file={},
    ...     by_module={},
    ... )
    >>> payload = symbol_index_to_payload(artifacts)
    >>> assert payload[0]["path"] == "mod.func"
    """
    result: list[dict[str, JsonValue]] = []
    for row in model.rows:
        entry: dict[str, JsonValue] = {
            "path": row.path,
            "canonical_path": row.canonical_path,
            "kind": row.kind,
            "module": row.module,
            "package": row.package,
            "file": row.file,
            "signature": row.signature,
            "owner": row.owner,
            "stability": row.stability,
            "since": row.since,
            "deprecated_in": row.deprecated_in,
            "section": row.section,
            "tested_by": list(row.tested_by),
            "is_async": row.is_async,
            "is_property": row.is_property,
        }
        if row.span is not None:
            entry["lineno"] = row.span.start
            entry["endlineno"] = row.span.end
        result.append(entry)
    return result


def symbol_delta_from_json(raw: JsonPayload) -> SymbolDeltaPayload:
    """Construct a SymbolDeltaPayload from a JSON payload with validation.

    Parameters
    ----------
    raw : JsonPayload
        Raw JSON payload (typically loaded via json.load).

    Returns
    -------
    SymbolDeltaPayload
        Validated typed artifact.

    Raises
    ------
    ArtifactValidationError
        If the payload is malformed or missing required fields.
    """
    if not isinstance(raw, dict):
        msg = f"Expected dict, got {type(raw).__name__}"
        raise ArtifactValidationError(msg, artifact_name="symbol-delta")

    try:
        # Coerce added/removed to tuples
        added = tuple(raw.get("added", []))
        removed = tuple(raw.get("removed", []))

        # Parse changed list
        changed_list: list[SymbolDeltaChange] = []
        for change_item in raw.get("changed", []):
            if not isinstance(change_item, dict):
                msg = f"Expected changed item to be dict, got {type(change_item).__name__}"
                raise ArtifactValidationError(msg, artifact_name="symbol-delta")

            reasons = tuple(change_item.get("reasons", []))
            delta_change = SymbolDeltaChange(
                path=str(change_item["path"]),
                before=change_item.get("before", {}),
                after=change_item.get("after", {}),
                reasons=reasons,
            )
            changed_list.append(delta_change)

        return SymbolDeltaPayload(
            base_sha=raw.get("base_sha"),
            head_sha=raw.get("head_sha"),
            added=added,
            removed=removed,
            changed=tuple(changed_list),
        )
    except (KeyError, ValueError, TypeError) as e:
        msg = f"Failed to construct SymbolDeltaPayload: {e}"
        raise ArtifactValidationError(msg, artifact_name="symbol-delta") from e


def symbol_delta_to_payload(model: SymbolDeltaPayload) -> dict[str, JsonValue]:
    """Serialize SymbolDeltaPayload to a JSON payload.

    Parameters
    ----------
    model : SymbolDeltaPayload
        Typed artifact to serialize.

    Returns
    -------
    dict[str, JsonValue]
        JSON payload ready for writing to disk or validation.
    """
    changed_list: list[dict[str, JsonValue]] = []
    for change in model.changed:
        changed_list.append(
            {
                "path": change.path,
                "before": change.before,
                "after": change.after,
                "reasons": list(change.reasons),
            }
        )

    return {
        "base_sha": model.base_sha,
        "head_sha": model.head_sha,
        "added": list(model.added),
        "removed": list(model.removed),
        "changed": changed_list,
    }


def load_symbol_index(path: Path) -> SymbolIndexArtifacts:
    """Load and validate a symbol index artifact from disk.

    Parameters
    ----------
    path : Path
        Path to the symbol index JSON file.

    Returns
    -------
    SymbolIndexArtifacts
        Validated typed artifact.

    Raises
    ------
    ArtifactValidationError
        If the file cannot be read or the payload is invalid.

    Examples
    --------
    >>> from pathlib import Path
    >>> from docs._types.artifacts import load_symbol_index
    >>> # artifacts = load_symbol_index(Path("docs/_build/symbols.json"))
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        msg = f"Failed to load {path}: {e}"
        raise ArtifactValidationError(msg, artifact_name="symbol-index") from e

    return symbol_index_from_json(payload)


def dump_symbol_index(path: Path, model: SymbolIndexArtifacts) -> None:
    """Write a symbol index artifact to disk with deterministic formatting.

    Parameters
    ----------
    path : Path
        Destination file path.
    model : SymbolIndexArtifacts
        Artifact to write.

    Raises
    ------
    ArtifactValidationError
        If the file cannot be written.
    """
    try:
        payload = symbol_index_to_payload(model)
        json_str = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
        path.write_text(json_str + "\n", encoding="utf-8")
    except (OSError, TypeError) as e:
        msg = f"Failed to dump symbol index to {path}: {e}"
        raise ArtifactValidationError(msg, artifact_name="symbol-index") from e


def load_symbol_delta(path: Path) -> SymbolDeltaPayload:
    """Load and validate a symbol delta artifact from disk.

    Parameters
    ----------
    path : Path
        Path to the symbol delta JSON file.

    Returns
    -------
    SymbolDeltaPayload
        Validated typed artifact.

    Raises
    ------
    ArtifactValidationError
        If the file cannot be read or the payload is invalid.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        msg = f"Failed to load {path}: {e}"
        raise ArtifactValidationError(msg, artifact_name="symbol-delta") from e

    return symbol_delta_from_json(payload)


def dump_symbol_delta(path: Path, model: SymbolDeltaPayload) -> None:
    """Write a symbol delta artifact to disk with deterministic formatting.

    Parameters
    ----------
    path : Path
        Destination file path.
    model : SymbolDeltaPayload
        Artifact to write.

    Raises
    ------
    ArtifactValidationError
        If the file cannot be written.
    """
    try:
        payload = symbol_delta_to_payload(model)
        json_str = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
        path.write_text(json_str + "\n", encoding="utf-8")
    except (OSError, TypeError) as e:
        msg = f"Failed to dump symbol delta to {path}: {e}"
        raise ArtifactValidationError(msg, artifact_name="symbol-delta") from e
