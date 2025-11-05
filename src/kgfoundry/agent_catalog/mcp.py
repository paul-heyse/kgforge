"""Stdio server implementing the Model Context Protocol for the Agent Catalog."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

from kgfoundry.agent_catalog.audit import AuditLogger
from kgfoundry.agent_catalog.cli import ALLOWED_FACET_KEYS, search_result_to_dict
from kgfoundry.agent_catalog.client import AgentCatalogClient, AgentCatalogClientError
from kgfoundry.agent_catalog.rbac import AccessController, Role
from kgfoundry_common.logging import (
    get_correlation_id,
    get_logger,
    set_correlation_id,
    with_fields,
)
from kgfoundry_common.observability import MetricsProvider, observe_duration
from kgfoundry_common.problem_details import JsonValue
from search_api.service import search_service

if TYPE_CHECKING:
    from search_api.types import VectorSearchResultTypedDict

logger = get_logger(__name__)

JsonObject = dict[str, JsonValue]

DEFAULT_CATALOG = Path("docs/_build/agent_catalog.json")
DEFAULT_AUDIT_LOG = Path("docs/_build/agent/audit.log")

INVALID_REQUEST_CODE = -32600
UNKNOWN_METHOD_CODE = -32601
INVALID_PARAMS_CODE = -32602
PARSE_ERROR_CODE = -32700
FORBIDDEN_CODE = -32000
CATALOG_ERROR_CODE = -32004
MIN_SEARCH_RESULTS = 1
MAX_SEARCH_RESULTS = 100


@dataclass(slots=True)
class RequestContext:
    """Context passed to JSON-RPC handlers.

    Contains request metadata including correlation ID and request ID for
    tracing and response matching in JSON-RPC over stdio.

    Parameters
    ----------
    correlation_id : str
        Correlation ID for request tracing and logging.
    request_id : JsonValue | None
        JSON-RPC request ID for matching responses. None for notifications.
    """

    correlation_id: str
    request_id: JsonValue | None


class CatalogSessionServerError(RuntimeError):
    """Raised when a JSON-RPC request cannot be fulfilled.

    Custom exception for JSON-RPC errors with HTTP status codes and
    Problem Details support. Used by the catalog session server to signal
    various error conditions.

    Parameters
    ----------
    status : int
        HTTP status code for the error (e.g., 400, 403, 404, 500).
    title : str
        Short error title (e.g., "invalid-params", "forbidden").
    detail : str
        Detailed error message describing the failure.
    code : int, optional
        JSON-RPC error code. Defaults to -32603 (internal error).
    """

    def __init__(self, status: int, title: str, detail: str, *, code: int = -32603) -> None:
        """Initialize JSON-RPC error.

        Creates a CatalogSessionServerError with the specified status, title,
        detail, and optional JSON-RPC error code.

        Parameters
        ----------
        status : int
            HTTP status code for the error.
        title : str
            Short error title.
        detail : str
            Detailed error message.
        code : int, optional
            JSON-RPC error code. Defaults to -32603.
        """
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail
        self.code = code

    def to_error(self, *, correlation_id: str | None = None) -> JsonObject:
        """Return JSON-RPC error object with Problem Details payload.

        Converts the exception to a JSON-RPC error response object with
        Problem Details structure in the data field.

        Parameters
        ----------
        correlation_id : str | None, optional
            Optional correlation ID to include in the Problem Details.
            Defaults to None.

        Returns
        -------
        JsonObject
            JSON-RPC error object with code, message, data (Problem Details),
            status, and detail fields.
        """
        problem: JsonObject = {
            "type": "https://kgfoundry.dev/problems/catalogctl-mcp",
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }
        if correlation_id:
            problem["correlation_id"] = correlation_id
        return {
            "code": self.code,
            "message": self.title,
            "data": problem,
            "status": self.status,
            "detail": self.detail,
        }


Handler = Callable[[JsonObject, RequestContext], JsonValue]


class CatalogSessionServer:
    """Minimal JSON-RPC server speaking over stdin/stdout.

    Implements a JSON-RPC 2.0 server that communicates over stdin/stdout
    for the Agent Catalog. Supports capabilities discovery, symbol lookup,
    search, and various catalog operations with RBAC and audit logging.

    Parameters
    ----------
    client : AgentCatalogClient
        Client for accessing the agent catalog.
    access : AccessController
        Access controller for RBAC enforcement.
    audit : AuditLogger
        Audit logger for recording operations.
    metrics : MetricsProvider | None, optional
        Optional metrics provider for observability. Defaults to None.
    """

    def __init__(
        self,
        client: AgentCatalogClient,
        *,
        access: AccessController,
        audit: AuditLogger,
        metrics: MetricsProvider | None = None,
    ) -> None:
        """Initialize the catalog session server.

        Creates a new CatalogSessionServer with the provided client, access
        controller, audit logger, and optional metrics provider.

        Parameters
        ----------
        client : AgentCatalogClient
            Client for accessing the agent catalog.
        access : AccessController
            Access controller for RBAC enforcement.
        audit : AuditLogger
            Audit logger for recording operations.
        metrics : MetricsProvider | None, optional
            Optional metrics provider. Defaults to None.
        """
        self.client = client
        self._shutdown = False
        self._access = access
        self._audit = audit
        self._role = access.role.value
        self._metrics = metrics or MetricsProvider.default()
        self._methods = cast(
            "dict[str, Handler]",
            {
                "initialize": self._handle_initialize,
                "catalog.capabilities": self._handle_capabilities,
                "catalog.symbol": self._handle_symbol,
                "catalog.find_callers": self._handle_find_callers,
                "catalog.find_callees": self._handle_find_callees,
                "catalog.change_impact": self._handle_change_impact,
                "catalog.suggest_tests": self._handle_suggest_tests,
                "catalog.open_anchor": self._handle_open_anchor,
                "catalog.search": self._handle_search,
                "catalog.list_modules": self._handle_list_modules,
                "session.shutdown": self._handle_shutdown,
                "session.exit": self._handle_exit,
            },
        )

    @staticmethod
    def _write(payload: JsonObject) -> None:
        """Write JSON-RPC payload to stdout.

        Writes a JSON-RPC message to stdout and flushes the buffer.
        Used for sending responses and errors to the client.

        Parameters
        ----------
        payload : JsonObject
            JSON-RPC payload object to write.
        """
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _respond(self, payload: JsonObject) -> None:
        """Send JSON-RPC response to client.

        Writes a JSON-RPC response message to stdout. Used for sending
        successful responses back to the client.

        Parameters
        ----------
        payload : JsonObject
            JSON-RPC response payload object to write.
        """
        self._write(payload)

    @staticmethod
    def _coerce_params(raw: object) -> JsonObject:
        """Coerce JSON-RPC params into a JSON object or raise an error.

        Validates and coerces JSON-RPC params to a dictionary. Raises
        CatalogSessionServerError if params is neither None nor a dict.

        Parameters
        ----------
        raw : object
            Raw params value from JSON-RPC request.

        Returns
        -------
        JsonObject
            Validated params dictionary (empty dict if None).

        Raises
        ------
        CatalogSessionServerError
            If params is not None and not a dict.
        """
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return cast("JsonObject", raw)
        message = "Params must be an object"
        raise CatalogSessionServerError(400, "invalid-params", message, code=INVALID_PARAMS_CODE)

    def _dispatch(
        self,
        method_raw: JsonValue,
        params: JsonObject,
        context: RequestContext,
    ) -> JsonValue:
        """Dispatch JSON-RPC method call to appropriate handler.

        Routes JSON-RPC method calls to registered handlers, enforces RBAC,
        logs operations, and handles errors. Includes structured logging
        and metrics observation.

        Parameters
        ----------
        method_raw : JsonValue
            Method name from JSON-RPC request.
        params : JsonObject
            Params dictionary from JSON-RPC request.
        context : RequestContext
            Request context with correlation ID and request ID.

        Returns
        -------
        JsonValue
            Method handler result.

        Raises
        ------
        CatalogSessionServerError
            If method is missing or invalid (400), unknown (404), or
            forbidden (403).
        """
        if not isinstance(method_raw, str):
            raise CatalogSessionServerError(
                400,
                "invalid-request",
                "Missing method name",
                code=INVALID_REQUEST_CODE,
            )
        handler = self._methods.get(method_raw)
        if handler is None:
            detail = f"Unknown method: {method_raw}"
            raise CatalogSessionServerError(404, "unknown-method", detail, code=UNKNOWN_METHOD_CODE)

        with (
            with_fields(
                logger,
                operation=method_raw,
                role=self._role,
                correlation_id=context.correlation_id,
            ) as log_adapter,
            observe_duration(
                self._metrics,
                method_raw,
                component="agent_catalog.mcp",
            ) as observer,
        ):
            log_adapter.debug("Dispatching JSON-RPC method", extra={"status": "started"})
            try:
                self._access.authorize(method_raw)
            except PermissionError as exc:
                self._audit.log(
                    action=method_raw,
                    role=self._role,
                    status="forbidden",
                    detail=str(exc),
                    correlation_id=context.correlation_id,
                )
                observer.error()
                raise CatalogSessionServerError(
                    403, "forbidden", str(exc), code=FORBIDDEN_CODE
                ) from exc

            try:
                result = handler(params, context)
            except CatalogSessionServerError:
                observer.error()
                raise
            except AgentCatalogClientError as exc:
                self._audit.log(
                    action=method_raw,
                    role=self._role,
                    status="catalog-error",
                    detail=str(exc),
                    correlation_id=context.correlation_id,
                )
                observer.error()
                raise CatalogSessionServerError(
                    404, "catalog-error", str(exc), code=CATALOG_ERROR_CODE
                ) from exc
            except (TypeError, ValueError) as exc:
                self._audit.log(
                    action=method_raw,
                    role=self._role,
                    status="invalid-params",
                    detail=str(exc),
                    correlation_id=context.correlation_id,
                )
                observer.error()
                raise CatalogSessionServerError(
                    400, "invalid-params", str(exc), code=INVALID_PARAMS_CODE
                ) from exc
            else:
                self._audit.log(
                    action=method_raw,
                    role=self._role,
                    status="ok",
                    correlation_id=context.correlation_id,
                )
                observer.success()
                log_adapter.info("JSON-RPC method completed", extra={"status": "success"})
                return result

    def _error(self, request_id: JsonValue | None, error: CatalogSessionServerError) -> None:
        """Send JSON-RPC error response to client.

        Writes a JSON-RPC error response to stdout with Problem Details
        payload. Logs the error with correlation ID for tracing.

        Parameters
        ----------
        request_id : JsonValue | None
            JSON-RPC request ID for matching the response. None for
            notifications or parse errors.
        error : CatalogSessionServerError
            Error exception to convert to JSON-RPC error response.
        """
        logger.debug(
            "JSON-RPC error %s: %s",
            error.status,
            error.detail,
            extra={"correlation_id": get_correlation_id()},
        )
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": error.to_error(correlation_id=get_correlation_id()),
            }
        )

    def serve(self) -> int:
        """Process JSON-RPC requests from stdin until shutdown is requested.

        Main server loop that reads JSON-RPC requests from stdin, dispatches
        them to handlers, and writes responses to stdout. Continues until
        shutdown is requested via shutdown or exit methods.

        Returns
        -------
        int
            Exit code (0 for normal shutdown).
        """
        stdin: TextIO = sys.stdin
        for raw_line in stdin:
            message_line = raw_line.strip()
            if not message_line:
                continue
            correlation_id = uuid.uuid4().hex
            set_correlation_id(correlation_id)
            try:
                message_raw: object = json.loads(message_line)
            except json.JSONDecodeError as exc:
                error = CatalogSessionServerError(
                    400, "invalid-json", str(exc), code=PARSE_ERROR_CODE
                )
                self._error(None, error)
                continue
            if not isinstance(message_raw, dict):
                error = CatalogSessionServerError(
                    400,
                    "invalid-request",
                    "JSON-RPC payload must be an object",
                    code=INVALID_REQUEST_CODE,
                )
                self._error(None, error)
                continue
            message_obj = cast("dict[str, object]", message_raw)
            params = self._coerce_params(message_obj.get("params"))
            method = cast("JsonValue", message_obj.get("method"))
            request_id = cast("JsonValue | None", message_obj.get("id"))
            context = RequestContext(
                correlation_id=correlation_id,
                request_id=request_id,
            )
            try:
                result = self._dispatch(method, params, context)
            except CatalogSessionServerError as exc:
                self._error(context.request_id, exc)
                continue
            if context.request_id is not None:
                self._respond({"jsonrpc": "2.0", "id": context.request_id, "result": result})
            if self._shutdown:
                break
        return 0

    def _handle_initialize(self, _params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle initialize JSON-RPC method.

        Returns server capabilities including available catalog procedures.
        Used by clients to discover available methods.

        Parameters
        ----------
        _params : JsonObject
            Method parameters (unused for initialize).
        _context : RequestContext
            Request context (unused for initialize).

        Returns
        -------
        JsonValue
            Capabilities object with procedures list.
        """
        commands = sorted(name for name in self._methods if name.startswith("catalog."))
        procedures = cast("list[JsonValue]", list(commands))
        capabilities: JsonObject = {"procedures": cast("JsonValue", procedures)}
        payload: JsonObject = {"capabilities": cast("JsonValue", capabilities)}
        return payload

    def _handle_capabilities(self, _params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle capabilities JSON-RPC method.

        Returns list of available package names from the catalog.

        Parameters
        ----------
        _params : JsonObject
            Method parameters (unused for capabilities).
        _context : RequestContext
            Request context (unused for capabilities).

        Returns
        -------
        JsonValue
            List of package names.
        """
        packages = [pkg.name for pkg in self.client.list_packages()]
        return cast("JsonValue", packages)

    def _handle_symbol(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle symbol JSON-RPC method.

        Retrieves symbol metadata by symbol ID from the catalog.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing symbol_id.
        _context : RequestContext
            Request context (unused for symbol lookup).

        Returns
        -------
        JsonValue
            Symbol metadata dictionary.

        Raises
        ------
        CatalogSessionServerError
            If symbol_id is not found (404).
        """
        symbol_id = str(params.get("symbol_id"))
        symbol = self.client.get_symbol(symbol_id)
        if symbol is None:
            detail = f"Unknown symbol: {symbol_id}"
            raise CatalogSessionServerError(404, "unknown-symbol", detail, code=CATALOG_ERROR_CODE)
        return cast("JsonValue", symbol.model_dump())

    def _handle_find_callers(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle find_callers JSON-RPC method.

        Finds all callers (references) of a symbol by symbol ID.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing symbol_id.
        _context : RequestContext
            Request context (unused for find_callers).

        Returns
        -------
        JsonValue
            List of caller symbol IDs.
        """
        symbol_id = str(params.get("symbol_id"))
        callers = self.client.find_callers(symbol_id)
        return cast("JsonValue", callers)

    def _handle_find_callees(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle find_callees JSON-RPC method.

        Finds all callees (called symbols) of a symbol by symbol ID.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing symbol_id.
        _context : RequestContext
            Request context (unused for find_callees).

        Returns
        -------
        JsonValue
            List of callee symbol IDs.
        """
        symbol_id = str(params.get("symbol_id"))
        callees = self.client.find_callees(symbol_id)
        return cast("JsonValue", callees)

    def _handle_change_impact(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle change_impact JSON-RPC method.

        Analyzes change impact for a symbol by symbol ID, returning
        affected symbols and modules.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing symbol_id.
        _context : RequestContext
            Request context (unused for change_impact).

        Returns
        -------
        JsonValue
            Change impact dictionary with affected symbols and modules.
        """
        symbol_id = str(params.get("symbol_id"))
        impact_raw = cast(
            "dict[str, JsonValue]",
            self.client.change_impact(symbol_id).model_dump(),
        )
        return cast("JsonValue", impact_raw)

    def _handle_suggest_tests(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle suggest_tests JSON-RPC method.

        Suggests test files and test functions for a symbol by symbol ID.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing symbol_id.
        _context : RequestContext
            Request context (unused for suggest_tests).

        Returns
        -------
        JsonValue
            List of test suggestion dictionaries.
        """
        symbol_id = str(params.get("symbol_id"))
        tests_raw: list[JsonValue] = []
        for test_entry in self.client.suggest_tests(symbol_id):
            converted_entry: JsonObject = {
                str(key): value for key, value in dict(test_entry).items()
            }
            tests_raw.append(converted_entry)
        return tests_raw

    def _handle_open_anchor(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle open_anchor JSON-RPC method.

        Returns anchor information (file path and line number) for a symbol
        by symbol ID, suitable for opening in an editor.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing symbol_id.
        _context : RequestContext
            Request context (unused for open_anchor).

        Returns
        -------
        JsonValue
            Anchor dictionary with file path and line number.
        """
        symbol_id = str(params.get("symbol_id"))
        anchor_raw = cast("JsonObject", self.client.open_anchor(symbol_id))
        return cast("JsonValue", anchor_raw)

    @staticmethod
    def _parse_limit(raw_k: JsonValue) -> int:
        """Validate and normalise the k parameter for search.

        Validates and coerces the k (limit) parameter for search operations.
        Accepts integers, float strings, or integer strings. Enforces
        bounds between MIN_SEARCH_RESULTS and MAX_SEARCH_RESULTS.

        Parameters
        ----------
        raw_k : JsonValue
            Raw k value from JSON-RPC params.

        Returns
        -------
        int
            Validated integer k value.

        Raises
        ------
        CatalogSessionServerError
            If k is not a valid integer or is out of range.
        """
        if isinstance(raw_k, bool):
            message = "k must be an integer"
            raise CatalogSessionServerError(
                400, "invalid-params", message, code=INVALID_PARAMS_CODE
            )
        if isinstance(raw_k, int):
            k_value = raw_k
        elif isinstance(raw_k, float):
            k_value = int(raw_k)
        elif isinstance(raw_k, str):
            stripped = raw_k.strip()
            if not stripped:
                message = "k must not be empty"
                raise CatalogSessionServerError(
                    400, "invalid-params", message, code=INVALID_PARAMS_CODE
                )
            try:
                k_value = int(stripped)
            except ValueError as exc:  # pragma: no cover - defensive guard
                message = "k must be an integer"
                raise CatalogSessionServerError(
                    400, "invalid-params", message, code=INVALID_PARAMS_CODE
                ) from exc
        else:
            message = "k must be an integer"
            raise CatalogSessionServerError(
                400, "invalid-params", message, code=INVALID_PARAMS_CODE
            )

        if not MIN_SEARCH_RESULTS <= k_value <= MAX_SEARCH_RESULTS:
            message = f"k must be between {MIN_SEARCH_RESULTS} and {MAX_SEARCH_RESULTS}"
            raise CatalogSessionServerError(
                400, "invalid-params", message, code=INVALID_PARAMS_CODE
            )
        return k_value

    @staticmethod
    def _parse_facets(raw_facets: JsonValue) -> dict[str, str]:
        """Validate facet filters and coerce values to strings.

        Validates and normalizes facet filters for search. Ensures only
        allowed facet keys are used and coerces values to strings.

        Parameters
        ----------
        raw_facets : JsonValue
            Raw facets value from JSON-RPC params.

        Returns
        -------
        dict[str, str]
            Validated facet dictionary with string values.

        Raises
        ------
        CatalogSessionServerError
            If facets is not None and not a dict, or if an invalid facet
            key is provided.
        """
        if raw_facets is None:
            return {}
        if not isinstance(raw_facets, dict):
            message = "facets must be an object"
            raise CatalogSessionServerError(
                400, "invalid-params", message, code=INVALID_PARAMS_CODE
            )
        facet_map: dict[str, str] = {}
        for key, value in raw_facets.items():
            if key not in ALLOWED_FACET_KEYS:
                allowed = ", ".join(sorted(ALLOWED_FACET_KEYS))
                message = f"Invalid facet key '{key}'. Allowed keys: {allowed}"
                raise CatalogSessionServerError(
                    400, "invalid-params", message, code=INVALID_PARAMS_CODE
                )
            if value is None:
                continue
            if key == "deprecated":
                facet_map[key] = "true" if bool(value) else "false"
            else:
                facet_map[key] = str(value).strip()
        return facet_map

    def _handle_search(self, params: JsonObject, context: RequestContext) -> JsonValue:
        """Handle search JSON-RPC method.

        Performs semantic search over the catalog with optional facet filters.
        Returns search results with metadata including query info and
        correlation ID.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing query, k (limit), and optional facets.
        context : RequestContext
            Request context for correlation ID tracking.

        Returns
        -------
        JsonValue
            Search response dictionary with results, total, took_ms, and
            metadata fields.

        Raises
        ------
        CatalogSessionServerError
            If query is missing, empty, or invalid (400).
        """
        raw_query = params.get("query", "")
        if not isinstance(raw_query, str):
            message = "Query must be a string"
            raise CatalogSessionServerError(
                400, "invalid-params", message, code=INVALID_PARAMS_CODE
            )
        query = raw_query.strip()
        if not query:
            message = "Query must not be empty"
            raise CatalogSessionServerError(
                400, "invalid-params", message, code=INVALID_PARAMS_CODE
            )

        k = self._parse_limit(params.get("k", 10))
        facet_map = self._parse_facets(params.get("facets"))

        results = self.client.search(query, k=k, facets=facet_map or None)
        typed_results = [
            cast("VectorSearchResultTypedDict", search_result_to_dict(result)) for result in results
        ]
        service_response = search_service(typed_results, metrics=self._metrics)
        metadata_obj: JsonObject = {
            str(key): value for key, value in service_response["metadata"].items()
        }
        query_info: JsonObject = {"query": query, "k": k}
        if facet_map:
            query_info["facets"] = dict(facet_map)
        metadata_obj.update(
            {
                "source": "agent_catalog.mcp",
                "correlation_id": context.correlation_id,
                "query_info": query_info,
            }
        )
        response_obj: JsonObject = {
            "results": cast("JsonValue", service_response["results"]),
            "total": service_response["total"],
            "took_ms": service_response["took_ms"],
            "metadata": metadata_obj,
        }
        response_obj["metadata"] = metadata_obj
        return response_obj

    def _handle_list_modules(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Handle list_modules JSON-RPC method.

        Lists all modules in a package by package name.

        Parameters
        ----------
        params : JsonObject
            Method parameters containing package name.
        _context : RequestContext
            Request context (unused for list_modules).

        Returns
        -------
        JsonValue
            List of qualified module names.
        """
        package = str(params.get("package"))
        modules = [module.qualified for module in self.client.list_modules(package)]
        return cast("JsonValue", modules)

    def _handle_shutdown(self, _params: JsonObject, _context: RequestContext) -> None:
        """Handle shutdown JSON-RPC method.

        Signals the server to shutdown gracefully after processing the
        current request.

        Parameters
        ----------
        _params : JsonObject
            Method parameters (unused for shutdown).
        _context : RequestContext
            Request context (unused for shutdown).
        """
        self._shutdown = True

    def _handle_exit(self, _params: JsonObject, _context: RequestContext) -> None:
        """Document  handle exit.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        _params : dict[str, object]
            Configure the  params.
        _context : RequestContext
            Configure the  context.
        """
        self._shutdown = True


def _resolve_role(value: str) -> Role:
    """Document  resolve role.

    <!-- auto:docstring-builder v1 -->

    &lt;!-- auto:docstring-builder v1 --&gt;

    Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

    Parameters
    ----------
    value : str
        Configure the value.

    Returns
    -------
    Role
        Describe return value.

    Raises
    ------
    ValueError
        Raised when message.
    """
    try:
        return Role(value)
    except ValueError as exc:
        valid = ", ".join(role.value for role in Role)
        message = f"Role must be one of: {valid}"
        raise ValueError(message) from exc


def build_parser() -> argparse.ArgumentParser:
    """Return an argument parser for the stdio server.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    argparse.ArgumentParser
        Describe return value.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to the agent catalog JSON artefact.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for resolving anchors.",
    )
    hosted_default = os.environ.get("CATALOG_HOSTED_MODE", "false").lower() in {"1", "true", "yes"}
    parser.add_argument(
        "--hosted-mode",
        action=argparse.BooleanOptionalAction,
        default=hosted_default,
        help="Enable RBAC and audit logging for hosted deployments.",
    )
    parser.add_argument(
        "--role",
        choices=[role.value for role in Role],
        default=os.environ.get("CATALOG_ROLE", Role.VIEWER.value),
        help="Role to apply when hosted mode is enabled.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=DEFAULT_AUDIT_LOG,
        help="Destination for audit JSONL entries when hosted mode is active.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the stdio session server.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    argv : list[str] | NoneType, optional
        Describe ``argv``.
        Defaults to ``None``.

    Returns
    -------
    int
        Describe return value.

    Raises
    ------
    SystemExit
        If catalog path or repository root does not exist, or if role is invalid.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog_path = cast("Path", args.catalog)
    if not catalog_path.exists():
        message = f"Catalog not found at {catalog_path}"
        raise SystemExit(message)
    repo_root = cast("Path", args.repo_root)
    if not repo_root.exists():
        message = f"Repository root does not exist: {repo_root}"
        raise SystemExit(message)
    client = AgentCatalogClient.from_path(catalog_path, repo_root=repo_root)
    try:
        role_value = cast("str", args.role)
        role = _resolve_role(role_value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    hosted_mode = cast("bool", args.hosted_mode)
    access = AccessController(role=role, enabled=hosted_mode)
    audit_log_path = cast("Path", args.audit_log)
    audit_logger = AuditLogger(audit_log_path, enabled=access.enabled)
    server = CatalogSessionServer(client, access=access, audit=audit_logger)
    return server.serve()


__all__ = ["CatalogSessionServer", "CatalogSessionServerError", "build_parser", "main"]
