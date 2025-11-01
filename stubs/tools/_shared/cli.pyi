# ruff: noqa: N803,N815
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools._shared.problem_details import ProblemDetailsDict

CliStatus = Literal["success", "violation", "config", "error"]
CliFileStatus = Literal["success", "skipped", "error", "violation"]
CliErrorStatus = Literal["error", "violation", "config"]

class CliFileResult:
    path: str
    status: CliFileStatus
    message: str | None
    problem: ProblemDetailsDict | None

    def __init__(
        self,
        *,
        path: str,
        status: CliFileStatus,
        message: str | None = ...,
        problem: ProblemDetailsDict | None = ...,
    ) -> None: ...

class CliErrorEntry:
    status: CliErrorStatus
    message: str
    file: str | None
    problem: ProblemDetailsDict | None

    def __init__(
        self,
        *,
        status: CliErrorStatus,
        message: str,
        file: str | None = ...,
        problem: ProblemDetailsDict | None = ...,
    ) -> None: ...

class CliEnvelope:
    schemaVersion: str
    schemaId: str
    generatedAt: str
    status: CliStatus
    command: str
    subcommand: str
    durationSeconds: float
    files: list[CliFileResult]
    errors: list[CliErrorEntry]
    problem: ProblemDetailsDict | None

    def __init__(
        self,
        *,
        schemaVersion: str = ...,
        schemaId: str = ...,
        generatedAt: str = ...,
        status: CliStatus = ...,
        command: str = ...,
        subcommand: str = ...,
        durationSeconds: float = ...,
        files: list[CliFileResult] | None = ...,
        errors: list[CliErrorEntry] | None = ...,
        problem: ProblemDetailsDict | None = ...,
    ) -> None: ...

@dataclass
class CliEnvelopeBuilder:
    envelope: CliEnvelope

    @classmethod
    def create(
        cls, *, command: str, status: CliStatus, subcommand: str = ""
    ) -> CliEnvelopeBuilder: ...
    def add_file(
        self,
        *,
        path: str,
        status: CliFileStatus,
        message: str | None = None,
        problem: ProblemDetailsDict | None = None,
    ) -> None: ...
    def add_error(
        self,
        *,
        status: CliErrorStatus,
        message: str,
        file: str | None = None,
        problem: ProblemDetailsDict | None = None,
    ) -> None: ...
    def set_problem(self, problem: ProblemDetailsDict | None) -> None: ...
    def finish(self, *, duration_seconds: float | None = None) -> CliEnvelope: ...

def new_cli_envelope(*, command: str, status: CliStatus, subcommand: str = "") -> CliEnvelope: ...
def validate_cli_envelope(envelope: CliEnvelope) -> None: ...
def render_cli_envelope(envelope: CliEnvelope, *, indent: int = 2) -> str: ...

CLI_ENVELOPE_SCHEMA: str
CLI_ENVELOPE_SCHEMA_ID: str
CLI_ENVELOPE_SCHEMA_VERSION: str
