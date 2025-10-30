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

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Final, Literal, TypedDict

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

CLI_SCHEMA_VERSION: Final = "1.0.0"
CLI_SCHEMA_ID: Final = "https://kgfoundry.dev/schema/docstring-builder-cli.json"

ParameterKind = Literal[
    "positional_only",
    "positional_or_keyword",
    "keyword_only",
    "var_positional",
    "var_keyword",
]

SymbolKind = Literal["function", "method", "class"]


class DocstringBuilderError(RuntimeError):
    """Base exception for docstring builder failures."""


class SchemaViolationError(DocstringBuilderError):
    """Raised when generated payloads do not satisfy the published schema."""


class PluginExecutionError(DocstringBuilderError):
    """Raised when a plugin fails during execution and cannot recover."""


class ToolConfigurationError(DocstringBuilderError):
    """Raised when CLI configuration (flags, environment, config files) is invalid."""


class RunStatus(str, Enum):
    """Enumerated status labels used across CLI summaries and manifest files."""

    SUCCESS = "success"
    VIOLATION = "violation"
    CONFIG = "config"
    ERROR = "error"


class ProblemDetails(TypedDict, total=False):
    """RFC 9457 Problem Details envelope used for error reporting."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    extensions: dict[str, JsonValue]


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
    problem: ProblemDetails


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
    }
    return payload


__all__ = [
    "CLI_SCHEMA_ID",
    "CLI_SCHEMA_VERSION",
    "PROBLEM_DETAILS_EXAMPLE",
    "DocfactsDocumentPayload",
    "DocfactsEntry",
    "DocfactsParameter",
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
    "InputHash",
    "JsonPrimitive",
    "JsonValue",
    "ParameterKind",
    "PluginExecutionError",
    "PluginReport",
    "PolicyReport",
    "PolicyViolationReport",
    "ProblemDetails",
    "RunStatus",
    "RunSummary",
    "SchemaViolationError",
    "StatusCounts",
    "ToolConfigurationError",
    "build_cli_result_skeleton",
]
