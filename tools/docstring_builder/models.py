"""Typed models and schemas for the docstring builder pipeline.

This module centralises the strongly typed intermediate representations used by the
docstring builder as well as the structures emitted through the CLI when the
``--json`` flag is enabled.  The definitions here are designed to remove the
dynamic ``dict[str, Any]`` payloads that currently trigger Ruff and mypy errors
throughout ``tools/docstring_builder``.  None of the existing modules import this
file yet; it establishes the target shapes that follow-up refactors will adopt.

Key goals addressed by these models:

* Typed DocFacts payloads aligned with ``docs/_build/schema_docfacts.json``.
* A versioned CLI result envelope accompanied by a JSON Schema (see
  ``schema/tools/docstring_builder_cli.json``).
* A small exception taxonomy that preserves error causality and surfaces RFC 9457
  Problem Details payloads for downstream tooling.
* Enumerations and literals that document permissible values, closing the gaps
  responsible for the observed mypy ``Any`` propagation and Ruff ``BLE001``/``T201``
  findings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict, cast

from jsonschema import Draft202012Validator, ValidationError

from tools.shared.problem_details import build_schema_problem_details

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

CLI_SCHEMA_VERSION: Final = "1.0.0"
CLI_SCHEMA_ID: Final = "https://kgfoundry.dev/schema/docstring-builder-cli.json"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCFACTS_SCHEMA_PATH = _REPO_ROOT / "docs" / "_build" / "schema_docfacts.json"
_CLI_SCHEMA_PATH = _REPO_ROOT / "schema" / "tools" / "docstring_builder_cli.json"


class ProblemDetails(TypedDict, total=False):
    """RFC 9457 Problem Details envelope used for error reporting."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    extensions: dict[str, JsonValue]


class DocstringBuilderError(RuntimeError):
    """Base exception for docstring builder failures."""


@dataclass(slots=True)
class SchemaViolationError(DocstringBuilderError):
    """Raised when generated payloads do not satisfy the published schema."""

    problem: ProblemDetails | None = None

    def __init__(self, message: str, *, problem: ProblemDetails | None = None) -> None:
        super().__init__(message)
        self.problem = problem


class PluginExecutionError(DocstringBuilderError):
    """Raised when a plugin fails during execution and cannot recover."""


class ToolConfigurationError(DocstringBuilderError):
    """Raised when CLI configuration (flags, environment, config files) is invalid."""


class SymbolResolutionError(DocstringBuilderError):
    """Raised when harvested symbols cannot be imported for inspection."""


class SignatureIntrospectionError(DocstringBuilderError):
    """Raised when callable signatures or annotations cannot be inspected."""


class RunStatus(StrEnum):
    """Enumerated status labels used across CLI summaries and manifest files."""

    SUCCESS = "success"
    VIOLATION = "violation"
    CONFIG = "config"
    ERROR = "error"


class DocfactsProvenanceLike(Protocol):
    """Structural typing contract for DocFacts provenance objects."""

    builder_version: str
    config_hash: str
    commit_hash: str
    generated_at: str


class DocFactLike(Protocol):
    """Structural view of DocFacts entries consumed by adapters."""

    qname: str
    module: str
    kind: str
    filepath: str
    lineno: int
    end_lineno: int | None
    decorators: Sequence[str]
    is_async: bool
    is_generator: bool
    owned: bool
    parameters: Sequence[Mapping[str, object]]
    returns: Sequence[Mapping[str, object]]
    raises: Sequence[Mapping[str, object]]
    notes: Sequence[str]


class DocfactsDocumentLike(Protocol):
    """Contract implemented by DocFacts document dataclasses."""

    docfacts_version: str
    provenance: DocfactsProvenanceLike
    entries: Sequence[DocFactLike]


class IRParameterLike(Protocol):
    """Minimal parameter fields required from legacy IR objects."""

    name: str
    display_name: str | None
    kind: str
    annotation: str | None
    default: str | None
    optional: bool
    description: str


class IRReturnLike(Protocol):
    """Minimal return metadata required from legacy IR objects."""

    kind: Literal["returns", "yields"]
    annotation: str | None
    description: str


class IRRaiseLike(Protocol):
    """Minimal raise metadata required from legacy IR objects."""

    exception: str
    description: str


class IRDocstringLike(Protocol):
    """Structural view of legacy docstring IR objects."""

    symbol_id: str
    module: str
    kind: str
    source_path: str
    lineno: int
    ir_version: str
    summary: str
    extended: str | None
    parameters: Sequence[IRParameterLike]
    returns: Sequence[IRReturnLike]
    raises: Sequence[IRRaiseLike]
    notes: Sequence[str]


ParameterKind = Literal[
    "positional_only",
    "positional_or_keyword",
    "keyword_only",
    "var_positional",
    "var_keyword",
]

_PARAMETER_KINDS: Final = {
    "positional_only",
    "positional_or_keyword",
    "keyword_only",
    "var_positional",
    "var_keyword",
}

SymbolKind = Literal["function", "method", "class"]

_SYMBOL_KINDS: Final = {"function", "method", "class"}

_RETURN_KINDS: Final = {"returns", "yields"}


PROBLEM_DETAILS_EXAMPLE: ProblemDetails = {
    "type": "https://kgfoundry.dev/problems/docbuilder/schema-mismatch",
    "title": "DocFacts schema validation failed",
    "status": 422,
    "detail": "Field anchors.endLine is missing",
    "instance": "urn:docbuilder:run:2025-10-30T12:00:00Z",
    "extensions": {
        "schemaVersion": "2.0.0",
        "docstringBuilderVersion": "1.6.0",
        "symbol": "kg.module.function",
    },
}

"""Example Problem Details payload conforming to RFC 9457.

Examples
--------
>>> from tools.docstring_builder.models import PROBLEM_DETAILS_EXAMPLE
>>> import json
>>> payload = json.dumps(PROBLEM_DETAILS_EXAMPLE, indent=2)
>>> assert "type" in payload
>>> assert "status" in payload
>>> assert PROBLEM_DETAILS_EXAMPLE["status"] == 422
>>> assert PROBLEM_DETAILS_EXAMPLE["type"].startswith("https://kgfoundry.dev/problems/")
"""


@dataclass(slots=True)
class DocstringIRParameter:
    """Typed representation of a parameter within the docstring IR."""

    name: str
    display_name: str
    kind: ParameterKind
    annotation: str | None = None
    default: str | None = None
    optional: bool = False
    description: str = ""


@dataclass(slots=True)
class DocstringIRReturn:
    """Typed representation of a return or yield section."""

    kind: Literal["returns", "yields"]
    annotation: str | None = None
    description: str = ""


@dataclass(slots=True)
class DocstringIRRaise:
    """Typed representation of an exception raised by the symbol."""

    exception: str
    description: str = ""


@dataclass(slots=True)
class DocstringIR:
    """Aggregate typed intermediate representation for a symbol's docstring."""

    symbol_id: str
    module: str
    kind: SymbolKind
    source_path: str
    lineno: int
    end_lineno: int | None
    ir_version: str
    summary: str
    extended: str | None = None
    parameters: list[DocstringIRParameter] = field(default_factory=list)
    returns: list[DocstringIRReturn] = field(default_factory=list)
    raises: list[DocstringIRRaise] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class DocfactsParameter(TypedDict, total=False):
    """JSON representation of a parameter preserved in DocFacts."""

    name: str
    display_name: str
    annotation: str | None
    optional: bool | None
    default: str | None
    kind: ParameterKind | None


class DocfactsReturn(TypedDict, total=False):
    """JSON representation of return metadata inside DocFacts."""

    kind: Literal["returns", "yields"]
    annotation: str | None
    description: str | None


class DocfactsRaise(TypedDict, total=False):
    """JSON representation of exception metadata inside DocFacts."""

    exception: str
    description: str | None


class DocfactsEntry(TypedDict, total=False):
    """Typed DocFacts entry aligned with ``schema_docfacts.json``."""

    qname: str
    module: str
    kind: SymbolKind
    filepath: str
    lineno: int
    end_lineno: int | None
    decorators: list[str]
    is_async: bool
    is_generator: bool
    owned: bool
    parameters: list[DocfactsParameter]
    returns: list[DocfactsReturn]
    raises: list[DocfactsRaise]
    notes: list[str]


class DocfactsProvenancePayload(TypedDict):
    """Provenance metadata stored alongside DocFacts outputs."""

    builderVersion: str
    configHash: str
    commitHash: str
    generatedAt: str


class DocfactsDocumentPayload(TypedDict):
    """Canonical DocFacts document payload."""

    docfactsVersion: str
    provenance: DocfactsProvenancePayload
    entries: list[DocfactsEntry]


class FileReport(TypedDict, total=False):
    """File-level result emitted by the CLI ``--json`` output."""

    path: str
    status: RunStatus
    changed: bool
    skipped: bool
    cacheHit: bool
    message: str
    preview: str
    baseline: str
    problem: ProblemDetails


class ErrorReport(TypedDict, total=False):
    """Summary of a failure encountered during processing."""

    file: str
    status: RunStatus
    message: str
    problem: ProblemDetails


class PolicyViolationReport(TypedDict):
    """Single policy engine violation entry."""

    rule: str
    symbol: str
    action: str
    message: str


class PolicyReport(TypedDict):
    """Aggregated policy engine results for a run."""

    coverage: float
    threshold: float
    violations: list[PolicyViolationReport]


class StatusCounts(TypedDict):
    """Mapping of :class:`RunStatus` to processed file counts."""

    success: int
    violation: int
    config: int
    error: int


class RunSummary(TypedDict, total=False):
    """Run summary statistics published in CLI output."""

    considered: int
    processed: int
    skipped: int
    changed: int
    status_counts: StatusCounts
    docfacts_checked: bool
    cache_hits: int
    cache_misses: int
    duration_seconds: float
    subcommand: str


class CacheSummary(TypedDict, total=False):
    """Cache metadata captured for observability and diagnostics."""

    path: str
    exists: bool
    hits: int
    misses: int
    mtime: str | None


class InputHash(TypedDict):
    """File hash metadata tracked for reproducibility."""

    hash: str
    mtime: str | None


class PluginReport(TypedDict):
    """Plugin execution overview included in CLI results."""

    enabled: list[str]
    available: list[str]
    disabled: list[str]
    skipped: list[str]


class DocfactsReport(TypedDict, total=False):
    """DocFacts summary embedded in CLI output."""

    path: str
    version: str
    diff: str | None
    validated: bool


class ObservabilityReport(TypedDict, total=False):
    """Observability payload mirrored in CLI results."""

    status: RunStatus
    errors: list[ErrorReport]
    driftPreviews: dict[str, str]


class CliResult(TypedDict, total=False):
    """Fully-typed structure matching ``schema/tools/docstring_builder_cli.json``."""

    schemaVersion: str
    schemaId: str
    status: RunStatus
    generatedAt: str
    command: str
    subcommand: str
    durationSeconds: float
    files: list[FileReport]
    errors: list[ErrorReport]
    summary: RunSummary
    policy: PolicyReport
    baseline: str
    cache: CacheSummary
    inputs: dict[str, InputHash]
    plugins: PluginReport
    docfacts: DocfactsReport
    observability: ObservabilityReport
    problem: ProblemDetails


_DOCFACTS_VALIDATOR: Final[Draft202012Validator] = Draft202012Validator(
    cast(dict[str, JsonValue], json.loads(_DOCFACTS_SCHEMA_PATH.read_text(encoding="utf-8")))
)
_CLI_VALIDATOR: Final[Draft202012Validator] = Draft202012Validator(
    cast(dict[str, JsonValue], json.loads(_CLI_SCHEMA_PATH.read_text(encoding="utf-8")))
)


def _load_docfacts_validator() -> Draft202012Validator:
    return _DOCFACTS_VALIDATOR


def _load_cli_validator() -> Draft202012Validator:
    return _CLI_VALIDATOR


def validate_docfacts_payload(payload: DocfactsDocumentPayload) -> None:
    """Validate ``payload`` against the published DocFacts schema."""
    validator = _load_docfacts_validator()
    try:
        validator.validate(payload)
    except ValidationError as exc:
        schema_version = payload.get("docfactsVersion")
        problem = build_schema_problem_details(
            error=exc,
            type="https://kgfoundry.dev/problems/docbuilder/docfacts-schema-mismatch",
            title="DocFacts schema validation failed",
            status=422,
            instance=f"urn:docbuilder:schema:docfacts:{datetime.now(tz=UTC).isoformat(timespec='seconds')}",
            extensions={"schemaVersion": schema_version or ""},
        )
        problem_payload = cast(ProblemDetails, problem)
        detail_message = str(problem_payload.get("detail", "DocFacts schema validation failed"))
        raise SchemaViolationError(detail_message, problem=problem_payload) from exc


def validate_cli_output(payload: CliResult) -> None:
    """Validate CLI machine output against the JSON schema."""
    validator = _load_cli_validator()
    try:
        validator.validate(payload)
    except ValidationError as exc:
        schema_version = payload.get("schemaVersion")
        problem = build_schema_problem_details(
            error=exc,
            type="https://kgfoundry.dev/problems/docbuilder/cli-schema-mismatch",
            title="CLI schema validation failed",
            status=422,
            instance=f"urn:docbuilder:schema:cli:{datetime.now(tz=UTC).isoformat(timespec='seconds')}",
            extensions={"schemaVersion": schema_version or ""},
        )
        problem_payload = cast(ProblemDetails, problem)
        detail_message = str(problem_payload.get("detail", "CLI schema validation failed"))
        raise SchemaViolationError(detail_message, problem=problem_payload) from exc


def _normalise_parameter(parameter: Mapping[str, object]) -> DocfactsParameter:
    name = str(parameter.get("name", ""))
    display_name = parameter.get("display_name")
    entry: DocfactsParameter = {
        "name": name,
        "display_name": str(display_name) if display_name is not None else name,
    }
    annotation = parameter.get("annotation")
    entry["annotation"] = str(annotation) if annotation is not None else None
    optional_flag = parameter.get("optional")
    entry["optional"] = bool(optional_flag) if optional_flag is not None else None
    default_value = parameter.get("default")
    entry["default"] = str(default_value) if default_value is not None else None
    kind_value = parameter.get("kind")
    if isinstance(kind_value, str) and kind_value in _PARAMETER_KINDS:
        entry["kind"] = cast(ParameterKind, kind_value)
    else:
        entry["kind"] = "positional_or_keyword"
    return entry


def _normalise_return(value: Mapping[str, object]) -> DocfactsReturn:
    kind_raw = str(value.get("kind", "returns"))
    entry: DocfactsReturn = {
        "kind": (
            cast(Literal["returns", "yields"], kind_raw) if kind_raw in _RETURN_KINDS else "returns"
        ),
    }
    annotation = value.get("annotation")
    entry["annotation"] = str(annotation) if annotation is not None else None
    description = value.get("description")
    entry["description"] = str(description) if description is not None else None
    return entry


def _normalise_raise(value: Mapping[str, object]) -> DocfactsRaise:
    entry: DocfactsRaise = {
        "exception": str(value.get("exception", "")),
    }
    description = value.get("description")
    entry["description"] = str(description) if description is not None else None
    return entry


def _build_docfacts_entry(fact: DocFactLike) -> DocfactsEntry:
    return {
        "qname": fact.qname,
        "module": fact.module,
        "kind": cast(SymbolKind, fact.kind) if fact.kind in _SYMBOL_KINDS else "function",
        "filepath": fact.filepath,
        "lineno": fact.lineno,
        "end_lineno": fact.end_lineno,
        "decorators": list(fact.decorators),
        "is_async": fact.is_async,
        "is_generator": fact.is_generator,
        "owned": fact.owned,
        "parameters": [_normalise_parameter(param) for param in fact.parameters],
        "returns": [_normalise_return(value) for value in fact.returns],
        "raises": [_normalise_raise(value) for value in fact.raises],
        "notes": list(fact.notes),
    }


def build_docfacts_document_payload(document: DocfactsDocumentLike) -> DocfactsDocumentPayload:
    """Convert a legacy :class:`DocfactsDocument` to the typed payload shape."""
    entries = [_build_docfacts_entry(fact) for fact in document.entries]
    provenance = document.provenance
    payload: DocfactsDocumentPayload = {
        "docfactsVersion": document.docfacts_version,
        "provenance": {
            "builderVersion": provenance.builder_version,
            "configHash": provenance.config_hash,
            "commitHash": provenance.commit_hash,
            "generatedAt": provenance.generated_at,
        },
        "entries": entries,
    }
    return payload


def build_docstring_ir_from_legacy(ir: IRDocstringLike) -> DocstringIR:
    """Convert a legacy IR object into the typed :class:`DocstringIR`."""
    parameters = [
        DocstringIRParameter(
            name=parameter.name,
            display_name=parameter.display_name or parameter.name,
            kind=_coerce_parameter_kind(parameter.kind),
            annotation=parameter.annotation,
            default=parameter.default,
            optional=parameter.optional,
            description=parameter.description,
        )
        for parameter in ir.parameters
    ]
    returns = [
        DocstringIRReturn(
            kind=ret.kind,
            annotation=ret.annotation,
            description=ret.description,
        )
        for ret in ir.returns
    ]
    raises = [
        DocstringIRRaise(exception=item.exception, description=item.description)
        for item in ir.raises
    ]
    return DocstringIR(
        symbol_id=ir.symbol_id,
        module=ir.module,
        kind=_coerce_symbol_kind(ir.kind),
        source_path=ir.source_path,
        lineno=ir.lineno,
        end_lineno=None,
        ir_version=ir.ir_version,
        summary=ir.summary,
        extended=ir.extended,
        parameters=parameters,
        returns=returns,
        raises=raises,
        notes=list(ir.notes),
    )


def build_cli_result_skeleton(status: RunStatus) -> CliResult:
    """Return a minimal CLI result payload initialised with required fields."""
    payload: CliResult = {
        "schemaVersion": CLI_SCHEMA_VERSION,
        "schemaId": CLI_SCHEMA_ID,
        "status": status,
        "generatedAt": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "files": [],
        "errors": [],
        "summary": {
            "considered": 0,
            "processed": 0,
            "skipped": 0,
            "changed": 0,
            "status_counts": {
                "success": 0,
                "violation": 0,
                "config": 0,
                "error": 0,
            },
            "docfacts_checked": False,
            "cache_hits": 0,
            "cache_misses": 0,
            "duration_seconds": 0.0,
            "subcommand": "",
        },
        "policy": {"coverage": 0.0, "threshold": 0.0, "violations": []},
        "cache": {"path": "", "exists": False, "hits": 0, "misses": 0, "mtime": None},
        "inputs": {},
        "plugins": {"enabled": [], "available": [], "disabled": [], "skipped": []},
        "observability": {"status": status, "errors": []},
    }
    return payload


def _json_pointer_from(error: ValidationError) -> str | None:
    raw_path = cast(Sequence[object], getattr(error, "absolute_path", ()))
    tokens: list[str] = []
    for part in raw_path:
        if part is None:
            continue
        tokens.append(str(part))
    if tokens:
        return "/" + "/".join(tokens)
    return None


def _coerce_parameter_kind(kind: str | None) -> ParameterKind:
    if kind in _PARAMETER_KINDS:
        return cast(ParameterKind, kind)
    return "positional_or_keyword"


def _coerce_symbol_kind(kind: str | None) -> SymbolKind:
    if kind in _SYMBOL_KINDS:
        return cast(SymbolKind, kind)
    return "function"


__all__ = sorted(
    [
        "CLI_SCHEMA_ID",
        "CLI_SCHEMA_VERSION",
        "CacheSummary",
        "DocFactLike",
        "DocfactsDocumentLike",
        "DocfactsDocumentPayload",
        "DocfactsEntry",
        "DocfactsParameter",
        "DocfactsProvenanceLike",
        "DocfactsProvenancePayload",
        "DocfactsRaise",
        "DocfactsReport",
        "DocfactsReturn",
        "DocstringBuilderError",
        "DocstringIR",
        "DocstringIRParameter",
        "DocstringIRRaise",
        "DocstringIRReturn",
        "ErrorReport",
        "FileReport",
        "IRDocstringLike",
        "IRParameterLike",
        "IRRaiseLike",
        "IRReturnLike",
        "InputHash",
        "PluginExecutionError",
        "PluginReport",
        "ObservabilityReport",
        "PolicyReport",
        "PolicyViolationReport",
        "PROBLEM_DETAILS_EXAMPLE",
        "ProblemDetails",
        "RunStatus",
        "RunSummary",
        "SchemaViolationError",
        "SignatureIntrospectionError",
        "StatusCounts",
        "SymbolResolutionError",
        "ToolConfigurationError",
        "build_cli_result_skeleton",
        "build_docfacts_document_payload",
        "build_docstring_ir_from_legacy",
        "validate_cli_output",
        "validate_docfacts_payload",
    ]
)
