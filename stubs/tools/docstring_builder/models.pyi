from collections.abc import Mapping
from typing import Any

DocstringBuilderError = Exception
SchemaViolationError = Exception
PluginExecutionError = Exception
ToolConfigurationError = Exception
SymbolResolutionError = Exception
SignatureIntrospectionError = Exception

ProblemDetails = Mapping[str, Any]

RunStatus = Any

DocFactLike = Any
DocfactsDocumentLike = Any
DocfactsDocumentPayload = Mapping[str, Any]
DocfactsEntry = Mapping[str, Any]
DocfactsParameter = Mapping[str, Any]
DocfactsProvenanceLike = Any
DocfactsProvenancePayload = Mapping[str, Any]
DocfactsRaise = Mapping[str, Any]
DocfactsReport = Mapping[str, Any]
DocfactsReturn = Mapping[str, Any]

DocstringIR = Any
DocstringIRParameter = Mapping[str, Any]
DocstringIRRaise = Mapping[str, Any]
DocstringIRReturn = Mapping[str, Any]

ErrorReport = Mapping[str, Any]
FileReport = Mapping[str, Any]
RunSummary = Mapping[str, Any]
CacheSummary = Mapping[str, Any]
InputHash = Mapping[str, Any]
PluginReport = Mapping[str, Any]
PolicyViolationReport = Mapping[str, Any]
PolicyReport = Mapping[str, Any]
StatusCounts = Mapping[str, Any]

PROBLEM_DETAILS_EXAMPLE: Mapping[str, Any]

build_docfacts_document_payload: Any
build_docstring_ir_from_legacy: Any
build_cli_result_skeleton: Any
validate_docfacts_payload: Any
validate_cli_output: Any

__all__ = tuple[str, ...]
