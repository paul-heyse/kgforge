"""Typing surface for the curated tooling API."""

# ruff: noqa: PLC2701, F401

from __future__ import annotations

import logging as _logging
from types import ModuleType
from typing import Final

from tools._shared.cli import (
    CLI_ENVELOPE_SCHEMA_ID,
    CLI_ENVELOPE_SCHEMA_VERSION,
    CliEnvelope,
    CliEnvelopeBuilder,
    CliErrorEntry,
    CliErrorStatus,
    CliFileResult,
    CliFileStatus,
    CliStatus,
    new_cli_envelope,
    render_cli_envelope,
    validate_cli_envelope,
)
from tools._shared.logging import (
    StructuredLoggerAdapter,
    get_logger,
    with_fields,
)
from tools._shared.metrics import (
    ToolRunObservation,
    ToolRunResult,
    observe_tool_run,
)
from tools._shared.problem_details import (
    ProblemDetailsDict,
    build_problem_details,
    build_schema_problem_details,
    build_tool_problem_details,
    problem_from_exception,
    render_problem,
    tool_disallowed_problem_details,
    tool_failure_problem_details,
    tool_missing_problem_details,
    tool_timeout_problem_details,
)
from tools._shared.proc import ToolExecutionError, run_tool
from tools._shared.schema import validate_tools_payload
from tools._shared.settings import (
    SettingsError,
    ToolRuntimeSettings,
    get_runtime_settings,
)
from tools._shared.validation import (
    ValidationError,
    require_directory,
    require_file,
    resolve_path,
)

logging = _logging

codemods: ModuleType
docstring_builder: ModuleType
docs: ModuleType
generate_pr_summary: ModuleType
navmap: ModuleType
build_agent_api: ModuleType
build_agent_catalog: ModuleType
gen_readmes: ModuleType

PUBLIC_EXPORTS: Final[dict[str, object]]

__all__ = sorted(
    [
        "CLI_ENVELOPE_SCHEMA_ID",
        "CLI_ENVELOPE_SCHEMA_VERSION",
        "CliEnvelope",
        "CliEnvelopeBuilder",
        "CliErrorEntry",
        "CliErrorStatus",
        "CliFileResult",
        "CliFileStatus",
        "CliStatus",
        "ProblemDetailsDict",
        "SettingsError",
        "StructuredLoggerAdapter",
        "ToolExecutionError",
        "ToolRunObservation",
        "ToolRunResult",
        "ToolRuntimeSettings",
        "ValidationError",
        "build_agent_api",
        "build_agent_catalog",
        "build_problem_details",
        "build_schema_problem_details",
        "build_tool_problem_details",
        "codemods",
        "docstring_builder",
        "docs",
        "gen_readmes",
        "generate_pr_summary",
        "get_logger",
        "get_runtime_settings",
        "logging",
        "navmap",
        "new_cli_envelope",
        "observe_tool_run",
        "problem_from_exception",
        "render_cli_envelope",
        "render_problem",
        "require_directory",
        "require_file",
        "resolve_path",
        "run_tool",
        "tool_disallowed_problem_details",
        "tool_failure_problem_details",
        "tool_missing_problem_details",
        "tool_timeout_problem_details",
        "validate_cli_envelope",
        "validate_tools_payload",
        "with_fields",
    ]
)
