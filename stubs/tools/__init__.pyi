"""Top-level package for internal tooling utilities."""

from __future__ import annotations

from typing import Any

CLI_ENVELOPE_SCHEMA_ID: str
CLI_ENVELOPE_SCHEMA_VERSION: str
CliEnvelope: Any
CliEnvelopeBuilder: Any
CliErrorEntry: Any
CliErrorStatus: Any
CliFileResult: Any
CliFileStatus: Any
CliStatus: Any
ProblemDetailsDict: Any
SettingsError: type[Exception]
StructuredLoggerAdapter: Any
ToolExecutionError: type[Exception]
ToolRunObservation: Any
ToolRunResult: Any
ToolRuntimeSettings: Any
ValidationError: type[Exception]
build_agent_api: Any
build_agent_catalog: Any
build_problem_details: Any
build_schema_problem_details: Any
build_tool_problem_details: Any
codemods: Any
docstring_builder: Any
docs: Any
gen_readmes: Any
generate_pr_summary: Any
get_logger: Any
get_runtime_settings: Any
logging: Any
navmap: Any
new_cli_envelope: Any
observe_tool_run: Any
problem_from_exception: Any
render_cli_envelope: Any
render_problem: Any
require_directory: Any
require_file: Any
resolve_path: Any
run_tool: Any
tool_disallowed_problem_details: Any
tool_failure_problem_details: Any
tool_missing_problem_details: Any
tool_timeout_problem_details: Any
validate_cli_envelope: Any
validate_tools_payload: Any
with_fields: Any

__all__: list[str]
