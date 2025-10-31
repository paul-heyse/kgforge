"""Top-level package for internal tooling utilities."""

from __future__ import annotations

import logging

from tools._shared import (
    CLI_ENVELOPE_SCHEMA_ID,
    CLI_ENVELOPE_SCHEMA_VERSION,
    CliEnvelope,
    CliEnvelopeBuilder,
    CliErrorEntry,
    CliErrorStatus,
    CliFileResult,
    CliFileStatus,
    CliStatus,
    ProblemDetailsDict,
    SettingsError,
    StructuredLoggerAdapter,
    ToolExecutionError,
    ToolRunObservation,
    ToolRunResult,
    ToolRuntimeSettings,
    build_problem_details,
    get_logger,
    get_runtime_settings,
    new_cli_envelope,
    observe_tool_run,
    render_cli_envelope,
    run_tool,
    validate_cli_envelope,
    validate_tools_payload,
    with_fields,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
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
    "build_problem_details",
    "get_logger",
    "get_runtime_settings",
    "logging",
    "new_cli_envelope",
    "observe_tool_run",
    "render_cli_envelope",
    "run_tool",
    "validate_cli_envelope",
    "validate_tools_payload",
    "with_fields",
]
