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
from typing import TextIO, cast

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

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    correlation_id : str
        Describe ``correlation_id``.
    request_id : object | NoneType
        Describe ``request_id``.
    """

    correlation_id: str
    request_id: JsonValue | None


class CatalogSessionServerError(RuntimeError):
    """Raised when a JSON-RPC request cannot be fulfilled.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    status : int
        Describe ``status``.
    title : str
        Describe ``title``.
    detail : str
        Describe ``detail``.
    code : int, optional
        Describe ``code``.
        Defaults to ``-32603``.
    """

    def __init__(self, status: int, title: str, detail: str, *, code: int = -32603) -> None:
        """Document   init  .

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        status : int
            Configure the status.
        title : str
            Configure the title.
        detail : str
            Configure the detail.
        code : int, optional
            Configure the code. Defaults to ``-32603``.
            Defaults to ``-32603``.
        """
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail
        self.code = code

    def to_error(self, *, correlation_id: str | None = None) -> JsonObject:
        """Return JSON-RPC error object with Problem Details payload.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        correlation_id : str | NoneType, optional
            Describe ``correlation_id``.
            Defaults to ``None``.

        Returns
        -------
        dict[str, object]
            Describe return value.
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

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    client : AgentCatalogClient
        Describe ``client``.
    access : AccessController
        Describe ``access``.
    audit : AuditLogger
        Describe ``audit``.
    metrics : MetricsProvider | None, optional
        Describe ``metrics``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        client: AgentCatalogClient,
        *,
        access: AccessController,
        audit: AuditLogger,
        metrics: MetricsProvider | None = None,
    ) -> None:
        """Document   init  .

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        client : AgentCatalogClient
            Configure the client.
        access : AccessController
            Configure the access.
        audit : AuditLogger
            Configure the audit.
        metrics : MetricsProvider | NoneType, optional
            Configure the metrics. Defaults to ``None``.
            Defaults to ``None``.
        """
        self.client = client
        self._shutdown = False
        self._access = access
        self._audit = audit
        self._role = access.role.value
        self._metrics = metrics or MetricsProvider.default()
        self._methods = cast(
            dict[str, Handler],
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
        """Document  write.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        payload : dict[str, object]
            Configure the payload.
        """
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _respond(self, payload: JsonObject) -> None:
        """Document  respond.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        payload : dict[str, object]
            Configure the payload.
        """
        self._write(payload)

    @staticmethod
    def _coerce_params(raw: object) -> JsonObject:
        """Coerce JSON-RPC params into a JSON object or raise an error.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        raw : object
            Describe ``raw``.

        Returns
        -------
        dict[str, object]
            Describe return value.
        """
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return cast(JsonObject, raw)
        message = "Params must be an object"
        raise CatalogSessionServerError(400, "invalid-params", message, code=INVALID_PARAMS_CODE)

    def _dispatch(
        self,
        method_raw: JsonValue,
        params: JsonObject,
        context: RequestContext,
    ) -> JsonValue:
        """Document  dispatch.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        method_raw : object
            Configure the method raw.
        params : dict[str, object]
            Configure the params.
        context : RequestContext
            Configure the context.

        Returns
        -------
        object
            Describe return value.

        Raises
        ------
        CatalogSessionServerError
            Raised when 400.
        CatalogSessionServerError
            Raised when 404.
        CatalogSessionServerError
            Raised when 403.
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
        """Document  error.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        request_id : object | NoneType
            Identifier for the request.
        error : CatalogSessionServerError
            Configure the error.
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

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        int
            Describe return value.
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
            message_obj = cast(dict[str, object], message_raw)
            params = self._coerce_params(message_obj.get("params"))
            method = cast(JsonValue, message_obj.get("method"))
            request_id = cast(JsonValue | None, message_obj.get("id"))
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
        """Document  handle initialize.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        _params : dict[str, object]
            Configure the  params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        commands = sorted(name for name in self._methods if name.startswith("catalog."))
        procedures = cast(list[JsonValue], list(commands))
        capabilities: JsonObject = {"procedures": cast(JsonValue, procedures)}
        payload: JsonObject = {"capabilities": cast(JsonValue, capabilities)}
        return payload

    def _handle_capabilities(self, _params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle capabilities.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        _params : dict[str, object]
            Configure the  params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        packages = [pkg.name for pkg in self.client.list_packages()]
        return cast(JsonValue, packages)

    def _handle_symbol(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle symbol.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.

        Raises
        ------
        CatalogSessionServerError
            Raised when 404.
        """
        symbol_id = str(params.get("symbol_id"))
        symbol = self.client.get_symbol(symbol_id)
        if symbol is None:
            detail = f"Unknown symbol: {symbol_id}"
            raise CatalogSessionServerError(404, "unknown-symbol", detail, code=CATALOG_ERROR_CODE)
        return cast(JsonValue, symbol.model_dump())

    def _handle_find_callers(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle find callers.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        symbol_id = str(params.get("symbol_id"))
        callers = self.client.find_callers(symbol_id)
        return cast(JsonValue, callers)

    def _handle_find_callees(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle find callees.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        symbol_id = str(params.get("symbol_id"))
        callees = self.client.find_callees(symbol_id)
        return cast(JsonValue, callees)

    def _handle_change_impact(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle change impact.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        symbol_id = str(params.get("symbol_id"))
        impact_raw = cast(
            dict[str, JsonValue],
            self.client.change_impact(symbol_id).model_dump(),
        )
        return cast(JsonValue, impact_raw)

    def _handle_suggest_tests(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle suggest tests.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
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
        """Document  handle open anchor.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        symbol_id = str(params.get("symbol_id"))
        anchor_raw = cast(JsonObject, self.client.open_anchor(symbol_id))
        return cast(JsonValue, anchor_raw)

    @staticmethod
    def _parse_limit(raw_k: JsonValue) -> int:
        """Validate and normalise the `k` parameter for search.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        raw_k : object
            Describe ``raw_k``.

        Returns
        -------
        int
            Describe return value.
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

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        raw_facets : object
            Describe ``raw_facets``.

        Returns
        -------
        dict[str, str]
            Describe return value.
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
        """Document  handle search.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        context : RequestContext
            Configure the context.

        Returns
        -------
        object
            Describe return value.

        Raises
        ------
        CatalogSessionServerError
            Raised when 400.
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
            cast(VectorSearchResultTypedDict, search_result_to_dict(result)) for result in results
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
            "results": cast(JsonValue, service_response["results"]),
            "total": service_response["total"],
            "took_ms": service_response["took_ms"],
            "metadata": metadata_obj,
        }
        response_obj["metadata"] = metadata_obj
        return response_obj

    def _handle_list_modules(self, params: JsonObject, _context: RequestContext) -> JsonValue:
        """Document  handle list modules.

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        params : dict[str, object]
            Configure the params.
        _context : RequestContext
            Configure the  context.

        Returns
        -------
        object
            Describe return value.
        """
        package = str(params.get("package"))
        modules = [module.qualified for module in self.client.list_modules(package)]
        return cast(JsonValue, modules)

    def _handle_shutdown(self, _params: JsonObject, _context: RequestContext) -> None:
        """Document  handle shutdown.

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
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog_path = cast(Path, args.catalog)
    if not catalog_path.exists():
        message = f"Catalog not found at {catalog_path}"
        raise SystemExit(message)
    repo_root = cast(Path, args.repo_root)
    if not repo_root.exists():
        message = f"Repository root does not exist: {repo_root}"
        raise SystemExit(message)
    client = AgentCatalogClient.from_path(catalog_path, repo_root=repo_root)
    try:
        role_value = cast(str, args.role)
        role = _resolve_role(role_value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    hosted_mode = cast(bool, args.hosted_mode)
    access = AccessController(role=role, enabled=hosted_mode)
    audit_log_path = cast(Path, args.audit_log)
    audit_logger = AuditLogger(audit_log_path, enabled=access.enabled)
    server = CatalogSessionServer(client, access=access, audit=audit_logger)
    return server.serve()


__all__ = ["CatalogSessionServer", "CatalogSessionServerError", "build_parser", "main"]
