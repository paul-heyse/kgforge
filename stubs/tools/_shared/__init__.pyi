from typing import Any

from tools._shared.problem_details import ProblemDetailsDict as _ProblemDetailsDict

ProblemDetailsDict = _ProblemDetailsDict

CLI_ENVELOPE_SCHEMA_ID: str
CLI_ENVELOPE_SCHEMA_VERSION: str

CliEnvelope = Any
CliEnvelopeBuilder = Any
CliErrorEntry = Any
CliErrorStatus = Any
CliFileResult = Any
CliFileStatus = Any
CliStatus = Any

SettingsError = type[Exception]
StructuredLoggerAdapter = Any
ToolExecutionError = type[Exception]
ToolRunObservation = Any
ToolRunResult = Any
ToolRuntimeSettings = Any
ValidationError = type[Exception]

run_tool: Any
observe_tool_run: Any
new_cli_envelope: Any
render_cli_envelope: Any
render_problem: Any
with_fields: Any

get_logger: Any
get_runtime_settings: Any
require_directory: Any
require_file: Any
resolve_path: Any
build_problem_details: Any
build_schema_problem_details: Any
build_tool_problem_details: Any
tool_disallowed_problem_details: Any
tool_failure_problem_details: Any
tool_missing_problem_details: Any
tool_timeout_problem_details: Any

validate_cli_envelope: Any
validate_tools_payload: Any
problem_from_exception: Any
