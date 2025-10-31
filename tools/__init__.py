"""Top-level package for internal tooling utilities."""

from __future__ import annotations

import logging

from tools import codemods as codemods
from tools import docs as docs
from tools import docstring_builder as docstring_builder
from tools import generate_pr_summary as generate_pr_summary
from tools import navmap as navmap
from tools._shared import (  # noqa: F401
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
    ValidationError,
    build_problem_details,
    build_schema_problem_details,
    build_tool_problem_details,
    get_logger,
    get_runtime_settings,
    new_cli_envelope,
    observe_tool_run,
    problem_from_exception,
    render_cli_envelope,
    render_problem,
    require_directory,
    require_file,
    resolve_path,
    run_tool,
    tool_disallowed_problem_details,
    tool_failure_problem_details,
    tool_missing_problem_details,
    tool_timeout_problem_details,
    validate_cli_envelope,
    validate_tools_payload,
    with_fields,
)
from tools.docs import build_agent_api as build_agent_api
from tools.docs import build_agent_catalog as build_agent_catalog
from tools.docs import gen_readmes as gen_readmes

logging.getLogger(__name__).addHandler(logging.NullHandler())

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
