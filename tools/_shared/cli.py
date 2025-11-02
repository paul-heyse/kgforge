# ruff: noqa: N815
"""Typed CLI envelope models and helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import msgspec
from msgspec import UNSET, Struct, UnsetType, structs

from tools._shared.problem_details import ProblemDetailsDict
from tools._shared.schema import validate_tools_payload

CliStatus = Literal["success", "violation", "config", "error"]
CliFileStatus = Literal["success", "skipped", "error", "violation"]
CliErrorStatus = Literal["error", "violation", "config"]

CLI_ENVELOPE_SCHEMA = "cli_envelope.json"
CLI_ENVELOPE_SCHEMA_VERSION = "1.0.0"
CLI_ENVELOPE_SCHEMA_ID = "https://kgfoundry.dev/schema/cli-envelope.json"


if TYPE_CHECKING:

    @dataclass(slots=True)
    class CliFileResult:
        """Individual file processing result emitted by tooling CLIs."""

        path: str
        status: CliFileStatus
        message: str | None = None
        problem: ProblemDetailsDict | UnsetType = UNSET

    @dataclass(slots=True)
    class CliErrorEntry:
        """Error-level entry attached to CLI envelopes."""

        status: CliErrorStatus
        message: str
        file: str | None = None
        problem: ProblemDetailsDict | UnsetType = UNSET

    @dataclass(slots=True)
    class CliEnvelope:
        """Typed representation of ``schema/tools/cli_envelope.json``."""

        schemaVersion: str = CLI_ENVELOPE_SCHEMA_VERSION
        schemaId: str = CLI_ENVELOPE_SCHEMA_ID
        generatedAt: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
        status: CliStatus = "success"
        command: str = ""
        subcommand: str = ""
        durationSeconds: float = 0.0
        files: list[CliFileResult] = field(default_factory=list)
        errors: list[CliErrorEntry] = field(default_factory=list)
        problem: ProblemDetailsDict | UnsetType = UNSET

else:

    class CliFileResult(Struct, kw_only=True):
        """Individual file processing result emitted by tooling CLIs."""

        path: str
        status: CliFileStatus
        message: str | None = None
        problem: ProblemDetailsDict | UnsetType = UNSET

    class CliErrorEntry(Struct, kw_only=True):
        """Error-level entry attached to CLI envelopes."""

        status: CliErrorStatus
        message: str
        file: str | None = None
        problem: ProblemDetailsDict | UnsetType = UNSET

    def _default_generated_at() -> str:
        return datetime.now(tz=UTC).isoformat()

    def _default_files() -> list[CliFileResult]:
        return []

    def _default_errors() -> list[CliErrorEntry]:
        return []

    class CliEnvelope(Struct, kw_only=True):
        """Typed representation of ``schema/tools/cli_envelope.json``."""

        schemaVersion: str = CLI_ENVELOPE_SCHEMA_VERSION
        schemaId: str = CLI_ENVELOPE_SCHEMA_ID
        generatedAt: str = msgspec.field(default_factory=_default_generated_at)
        status: CliStatus = "success"
        command: str = ""
        subcommand: str = ""
        durationSeconds: float = 0.0
        files: list[CliFileResult] = msgspec.field(default_factory=_default_files)
        errors: list[CliErrorEntry] = msgspec.field(default_factory=_default_errors)
        problem: ProblemDetailsDict | UnsetType = UNSET


def new_cli_envelope(*, command: str, status: CliStatus, subcommand: str = "") -> CliEnvelope:
    """Return a freshly initialised CLI envelope.

    Returns
    -------
    CliEnvelope
        Envelope populated with the required metadata for ``status``.
    """
    return CliEnvelope(command=command, status=status, subcommand=subcommand)


def validate_cli_envelope(envelope: CliEnvelope) -> None:
    """Validate ``envelope`` against the CLI schema.

    Returns
    -------
    None
        This function raises if validation fails.
    """
    payload: dict[str, object] = msgspec.to_builtins(envelope)
    validate_tools_payload(payload, CLI_ENVELOPE_SCHEMA)


def render_cli_envelope(envelope: CliEnvelope, *, indent: int = 2) -> str:
    """Return a JSON string for ``envelope``."""
    payload: dict[str, object] = msgspec.to_builtins(envelope)
    return json.dumps(payload, indent=indent)


@dataclass(slots=True)
class CliEnvelopeBuilder:
    """Mutable builder for assembling CLI envelopes."""

    envelope: CliEnvelope

    @classmethod
    def create(cls, *, command: str, status: CliStatus, subcommand: str = "") -> CliEnvelopeBuilder:
        return cls(new_cli_envelope(command=command, status=status, subcommand=subcommand))

    def add_file(
        self,
        *,
        path: str,
        status: CliFileStatus,
        message: str | None = None,
        problem: ProblemDetailsDict | None = None,
    ) -> None:
        self.envelope.files.append(
            CliFileResult(
                path=path,
                status=status,
                message=message,
                problem=problem if problem is not None else UNSET,
            )
        )

    def add_error(
        self,
        *,
        status: CliErrorStatus,
        message: str,
        file: str | None = None,
        problem: ProblemDetailsDict | None = None,
    ) -> None:
        self.envelope.errors.append(
            CliErrorEntry(
                status=status,
                message=message,
                file=file,
                problem=problem if problem is not None else UNSET,
            )
        )

    def set_problem(self, problem: ProblemDetailsDict | None) -> None:
        replacement: ProblemDetailsDict | UnsetType = problem if problem is not None else UNSET
        self.envelope = structs.replace(self.envelope, problem=replacement)

    def finish(self, *, duration_seconds: float | None = None) -> CliEnvelope:
        if duration_seconds is not None:
            self.envelope = structs.replace(self.envelope, durationSeconds=float(duration_seconds))
        validate_cli_envelope(self.envelope)
        return self.envelope


__all__ = [
    "CLI_ENVELOPE_SCHEMA",
    "CLI_ENVELOPE_SCHEMA_ID",
    "CLI_ENVELOPE_SCHEMA_VERSION",
    "CliEnvelope",
    "CliEnvelopeBuilder",
    "CliErrorEntry",
    "CliErrorStatus",
    "CliFileResult",
    "CliFileStatus",
    "CliStatus",
    "new_cli_envelope",
    "render_cli_envelope",
    "validate_cli_envelope",
]
