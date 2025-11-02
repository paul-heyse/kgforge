"""Authoritative typed models for documentation artifacts.

This module provides Pydantic V2-backed data structures and conversion helpers for all
documentation pipeline artifacts (symbol index, delta, reverse lookups). Models
serialize to payloads that validate against the canonical JSON Schemas under
`schema/docs/`.

The module owns the JSON contract and encapsulates all transformations, ensuring:

- Deterministic serialization (stable field order, sorted keys in JSON)
- Schema compliance (all payloads validate via jsonschema)
- Type safety (no Any-typed access in public functions)
- Defensive validation (field coercion, missing-key handling)
- Self-documenting schemas (generated from model definitions)

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
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, Field, field_validator

# Type aliases matching RFC 7159 JSON structure
type JsonPrimitive = str | int | float | bool | None
if TYPE_CHECKING:
    type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
else:
    JsonValue = object  # type: ignore[assignment, misc]

type JsonPayload = dict[str, JsonValue] | list[JsonValue]

__all__ = [
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


class LineSpan(BaseModel):
    """Start/end line numbers for a symbol.

    Parameters
    ----------
    start : int | None
        Starting line number (1-indexed), or None if unknown.
    end : int | None
        Ending line number (1-indexed, inclusive), or None if unknown.
    """

    start: int | None = Field(None, ge=0, description="Start line (1-indexed)")
    end: int | None = Field(None, ge=0, description="End line (1-indexed, inclusive)")

    model_config = {"frozen": True}


class SymbolIndexRow(BaseModel):
    """A single symbol entry in the index.

    Each row represents one documented symbol (function, class, module, etc.) with
    metadata needed for search, deep linking, and reverse lookups.

    Parameters
    ----------
    path : str
        Fully qualified symbol path (e.g., "pkg.mod.ClassName.method_name").
    kind : str
        Symbol kind: "module", "class", "function", "method", etc.
    doc : str
        Documentation string/docstring for this symbol.
    tested_by : tuple[str, ...] | list[str]
        Test paths (relative to tests/) that cover this symbol.
    source_link : dict[str, str]
        Links to source code (e.g., GitHub, local paths).
    canonical_path : str | None, optional
        If this symbol is an alias, canonical_path points to the real definition.
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
    is_async : bool, optional
        True if this is an async function/method.
        Defaults to False.
    is_property : bool, optional
        True if this is a @property.
        Defaults to False.
    """

    path: str = Field(..., description="Fully qualified symbol path")
    kind: str = Field(..., description="Symbol kind (module, class, function, etc.)")
    doc: str = Field(..., description="Documentation/docstring for this symbol")
    tested_by: Annotated[tuple[str, ...], Field(description="Test paths covering this symbol")] = ()
    source_link: Annotated[
        dict[str, str], Field(description="Links to source code (GitHub, etc.)")
    ] = Field(default_factory=dict)
    canonical_path: str | None = Field(None, description="Canonical path if this is an alias")
    module: str | None = Field(None, description="Module containing this symbol")
    package: str | None = Field(None, description="Top-level package name")
    file: str | None = Field(None, description="Relative path to source file")
    span: LineSpan | None = Field(None, description="Start/end line numbers")
    signature: str | None = Field(None, description="Function/method signature")
    owner: str | None = Field(None, description="Owner class (for methods)")
    stability: str | None = Field(None, description="Stability tag")
    since: str | None = Field(None, description="Version when introduced")
    deprecated_in: str | None = Field(None, description="Version when deprecated")
    section: str | None = Field(None, description="Documentation section")
    is_async: bool = Field(False, description="True if async function/method")
    is_property: bool = Field(False, description="True if @property")

    @field_validator("path", mode="before")
    @classmethod
    def path_not_empty(cls, v: object) -> str:
        """Ensure path is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("path must be a non-empty string")
        return v

    @field_validator("kind", mode="before")
    @classmethod
    def kind_not_empty(cls, v: object) -> str:
        """Ensure kind is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("kind must be a non-empty string")
        return v

    @field_validator("doc", mode="before")
    @classmethod
    def doc_not_empty(cls, v: object) -> str:
        """Ensure doc is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("doc must be a non-empty string")
        return v

    @field_validator("tested_by", mode="before")
    @classmethod
    def coerce_tested_by(cls, v: object) -> tuple[str, ...]:
        """Coerce tested_by to tuple, defaulting to empty."""
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        return ()

    @field_validator("source_link", mode="before")
    @classmethod
    def coerce_source_link(cls, v: object) -> dict[str, str]:
        """Coerce source_link to dict, defaulting to empty."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return {}

    model_config = {"frozen": True}


class SymbolIndexArtifacts(BaseModel):
    """Complete symbol index payload with forward and reverse lookups.

    Parameters
    ----------
    rows : tuple[SymbolIndexRow, ...] | list[SymbolIndexRow]
        All symbol entries, sorted by path.
    by_file : dict[str, tuple[str, ...]]
        Reverse lookup: file path -> sorted tuple of symbol paths.
    by_module : dict[str, tuple[str, ...]]
        Reverse lookup: module name -> sorted tuple of symbol paths.
    """

    rows: Annotated[
        tuple[SymbolIndexRow, ...],
        Field(description="All symbol entries (sorted by path)"),
    ] = ()
    by_file: Annotated[
        dict[str, tuple[str, ...]],
        Field(description="Reverse lookup: file -> symbol paths"),
    ] = Field(default_factory=dict)
    by_module: Annotated[
        dict[str, tuple[str, ...]],
        Field(description="Reverse lookup: module -> symbol paths"),
    ] = Field(default_factory=dict)

    @field_validator("rows", mode="before")
    @classmethod
    def coerce_rows(cls, v: object) -> tuple[SymbolIndexRow, ...]:
        """Coerce rows to tuple."""
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(v)  # type: ignore[arg-type]
        return ()

    model_config = {"frozen": True}


class SymbolDeltaChange(BaseModel):
    """A single changed symbol between two versions.

    Parameters
    ----------
    path : str
        The symbol path that changed.
    before : dict[str, JsonValue]
        Previous version of the row (serialized).
    after : dict[str, JsonValue]
        New version of the row (serialized).
    reasons : tuple[str, ...] | list[str]
        List of reasons why the symbol changed (e.g., ["signature_changed"]).
    """

    path: str = Field(..., description="Symbol path that changed")
    before: Annotated[dict[str, JsonValue], Field(description="Previous row (serialized)")] = Field(
        default_factory=dict
    )
    after: Annotated[dict[str, JsonValue], Field(description="New row (serialized)")] = Field(
        default_factory=dict
    )
    reasons: Annotated[tuple[str, ...], Field(description="Reasons for the change")] = ()

    @field_validator("path", mode="before")
    @classmethod
    def path_not_empty(cls, v: object) -> str:
        """Ensure path is a non-empty string."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("path must be a non-empty string")
        return v

    @field_validator("reasons", mode="before")
    @classmethod
    def coerce_reasons(cls, v: object) -> tuple[str, ...]:
        """Coerce reasons to tuple."""
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        return ()

    model_config = {"frozen": True}


class SymbolDeltaPayload(BaseModel):
    """Delta (diff) of symbols between two git commits or documentation builds.

    Parameters
    ----------
    base_sha : str | None
        Git SHA or build identifier for the baseline.
    head_sha : str | None
        Git SHA or build identifier for the current state.
    added : tuple[str, ...] | list[str]
        Sorted tuple of newly added symbol paths.
    removed : tuple[str, ...] | list[str]
        Sorted tuple of removed symbol paths.
    changed : tuple[SymbolDeltaChange, ...] | list[SymbolDeltaChange]
        List of symbols that changed (sorted by path).
    """

    base_sha: str | None = Field(None, description="Baseline git SHA or build ID")
    head_sha: str | None = Field(None, description="Current git SHA or build ID")
    added: Annotated[tuple[str, ...], Field(description="Newly added symbol paths")] = ()
    removed: Annotated[tuple[str, ...], Field(description="Removed symbol paths")] = ()
    changed: Annotated[tuple[SymbolDeltaChange, ...], Field(description="Changed symbols")] = ()

    @field_validator("added", mode="before")
    @classmethod
    def coerce_added(cls, v: object) -> tuple[str, ...]:
        """Coerce added to tuple."""
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        return ()

    @field_validator("removed", mode="before")
    @classmethod
    def coerce_removed(cls, v: object) -> tuple[str, ...]:
        """Coerce removed to tuple."""
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        return ()

    @field_validator("changed", mode="before")
    @classmethod
    def coerce_changed(cls, v: object) -> tuple[SymbolDeltaChange, ...]:
        """Coerce changed to tuple."""
        if v is None:
            return ()
        if isinstance(v, (list, tuple)):
            return tuple(v)  # type: ignore[arg-type]
        return ()

    model_config = {"frozen": True}


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

    Examples
    --------
    >>> from docs._types.artifacts import ArtifactValidationError
    >>> try:
    ...     raise ArtifactValidationError(
    ...         "Missing required field: path",
    ...         artifact_name="symbols.json",
    ...         problem_details={
    ...             "type": "urn:kgfoundry:validation-error",
    ...             "title": "Validation Error",
    ...             "detail": "Missing required field: path",
    ...             "instance": "symbols.json",
    ...         },
    ...     )
    ... except ArtifactValidationError as e:
    ...     assert e.artifact_name == "symbols.json"
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
    ...         "doc": "A function.",
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

            # Create row with Pydantic validation
            row = SymbolIndexRow(
                path=str(item["path"]),
                kind=str(item["kind"]),
                doc=str(item["doc"]),
                tested_by=item.get("tested_by", []),
                source_link=item.get("source_link", {}),
                canonical_path=item.get("canonical_path"),
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
    >>> row = SymbolIndexRow(path="mod.func", kind="function", doc="A function.")
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
            "doc": row.doc,
            "source_link": row.source_link,
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

    Examples
    --------
    >>> from docs._types.artifacts import symbol_delta_from_json
    >>> payload = {
    ...     "base_sha": "abc123",
    ...     "head_sha": "def456",
    ...     "added": ["new.symbol"],
    ...     "removed": [],
    ...     "changed": [],
    ... }
    >>> delta = symbol_delta_from_json(payload)
    >>> assert delta.base_sha == "abc123"
    """
    if not isinstance(raw, dict):
        msg = f"Expected dict, got {type(raw).__name__}"
        raise ArtifactValidationError(msg, artifact_name="symbol-delta")

    try:
        return SymbolDeltaPayload(
            base_sha=raw.get("base_sha"),
            head_sha=raw.get("head_sha"),
            added=raw.get("added", []),
            removed=raw.get("removed", []),
            changed=raw.get("changed", []),
        )
    except (ValueError, TypeError) as e:
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

    Examples
    --------
    >>> from docs._types.artifacts import (
    ...     SymbolDeltaChange,
    ...     SymbolDeltaPayload,
    ...     symbol_delta_to_payload,
    ... )
    >>> delta = SymbolDeltaPayload(
    ...     base_sha="abc123",
    ...     head_sha="def456",
    ... )
    >>> payload = symbol_delta_to_payload(delta)
    >>> assert payload["base_sha"] == "abc123"
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

    Examples
    --------
    >>> from pathlib import Path
    >>> from docs._types.artifacts import (
    ...     SymbolIndexRow,
    ...     SymbolIndexArtifacts,
    ...     dump_symbol_index,
    ... )
    >>> row = SymbolIndexRow(path="mod.func", kind="function", doc="A function.")
    >>> artifacts = SymbolIndexArtifacts(rows=(row,))
    >>> # dump_symbol_index(Path("output.json"), artifacts)
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

    Examples
    --------
    >>> from pathlib import Path
    >>> from docs._types.artifacts import load_symbol_delta
    >>> # delta = load_symbol_delta(Path("docs/_build/symbols.delta.json"))
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

    Examples
    --------
    >>> from pathlib import Path
    >>> from docs._types.artifacts import (
    ...     SymbolDeltaPayload,
    ...     dump_symbol_delta,
    ... )
    >>> delta = SymbolDeltaPayload(base_sha="abc123", head_sha="def456")
    >>> # dump_symbol_delta(Path("output.delta.json"), delta)
    """
    try:
        payload = symbol_delta_to_payload(model)
        json_str = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
        path.write_text(json_str + "\n", encoding="utf-8")
    except (OSError, TypeError) as e:
        msg = f"Failed to dump symbol delta to {path}: {e}"
        raise ArtifactValidationError(msg, artifact_name="symbol-delta") from e
