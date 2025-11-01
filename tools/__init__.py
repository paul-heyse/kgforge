"""Curated public API for KGFoundry tooling.

The exports in this module provide stable entry points for automation. Runtime failures
follow :mod:`kgfoundry_common.errors` and emit Problem Details envelopes consistent with
``schema/examples/tools/problem_details/tool-execution-error.json``.
"""

# ruff: noqa: TID252

from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType
from typing import Final

from ._shared.cli import (
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
from ._shared.logging import (
    StructuredLoggerAdapter,
    get_logger,
    with_fields,
)
from ._shared.metrics import ToolRunObservation, observe_tool_run
from ._shared.problem_details import (
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
from ._shared.proc import ToolExecutionError, ToolRunResult, run_tool
from ._shared.schema import validate_tools_payload
from ._shared.settings import (
    SettingsError,
    ToolRuntimeSettings,
    get_runtime_settings,
)
from ._shared.validation import (
    ValidationError,
    require_directory,
    require_file,
    resolve_path,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

PUBLIC_EXPORTS: Final[dict[str, object]] = {
    "CLI_ENVELOPE_SCHEMA_ID": CLI_ENVELOPE_SCHEMA_ID,
    "CLI_ENVELOPE_SCHEMA_VERSION": CLI_ENVELOPE_SCHEMA_VERSION,
    "CliEnvelope": CliEnvelope,
    "CliEnvelopeBuilder": CliEnvelopeBuilder,
    "CliErrorEntry": CliErrorEntry,
    "CliErrorStatus": CliErrorStatus,
    "CliFileResult": CliFileResult,
    "CliFileStatus": CliFileStatus,
    "CliStatus": CliStatus,
    "ProblemDetailsDict": ProblemDetailsDict,
    "SettingsError": SettingsError,
    "StructuredLoggerAdapter": StructuredLoggerAdapter,
    "ToolExecutionError": ToolExecutionError,
    "ToolRunObservation": ToolRunObservation,
    "ToolRunResult": ToolRunResult,
    "ToolRuntimeSettings": ToolRuntimeSettings,
    "ValidationError": ValidationError,
    "build_problem_details": build_problem_details,
    "build_schema_problem_details": build_schema_problem_details,
    "build_tool_problem_details": build_tool_problem_details,
    "get_logger": get_logger,
    "get_runtime_settings": get_runtime_settings,
    "logging": logging,
    "new_cli_envelope": new_cli_envelope,
    "observe_tool_run": observe_tool_run,
    "problem_from_exception": problem_from_exception,
    "render_cli_envelope": render_cli_envelope,
    "render_problem": render_problem,
    "require_directory": require_directory,
    "require_file": require_file,
    "resolve_path": resolve_path,
    "run_tool": run_tool,
    "tool_disallowed_problem_details": tool_disallowed_problem_details,
    "tool_failure_problem_details": tool_failure_problem_details,
    "tool_missing_problem_details": tool_missing_problem_details,
    "tool_timeout_problem_details": tool_timeout_problem_details,
    "validate_cli_envelope": validate_cli_envelope,
    "validate_tools_payload": validate_tools_payload,
    "with_fields": with_fields,
}

MODULE_EXPORTS: Final[dict[str, str]] = {
    "build_agent_api": "tools.docs.build_agent_api",
    "build_agent_catalog": "tools.docs.build_agent_catalog",
    "codemods": "tools.codemods",
    "docstring_builder": "tools.docstring_builder",
    "docs": "tools.docs",
    "gen_readmes": "tools.gen_readmes",
    "generate_pr_summary": "tools.generate_pr_summary",
    "navmap": "tools.navmap",
}

__all__: list[str] = sorted({*PUBLIC_EXPORTS.keys(), *MODULE_EXPORTS.keys()})


def __getattr__(name: str) -> ModuleType:
    if name in MODULE_EXPORTS:
        module = import_module(MODULE_EXPORTS[name])
        globals()[name] = module
        PUBLIC_EXPORTS[name] = module
        return module
    message = f"module 'tools' has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted({*globals(), *MODULE_EXPORTS.keys()})
