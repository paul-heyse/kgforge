"""Stdio server implementing the Model Context Protocol for the Agent Catalog."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from kgfoundry.agent_catalog.audit import AuditLogger
from kgfoundry.agent_catalog.client import AgentCatalogClient, AgentCatalogClientError
from kgfoundry.agent_catalog.rbac import AccessController, Role

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

DEFAULT_CATALOG = Path("docs/_build/agent_catalog.json")
DEFAULT_AUDIT_LOG = Path("docs/_build/agent/audit.log")


def build_parser() -> argparse.ArgumentParser:
    """Return an argument parser for the stdio server."""
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


class CatalogSessionServerError(RuntimeError):
    """Raised when a JSON-RPC request cannot be fulfilled."""

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail

    def to_problem(self) -> JsonObject:
        """Return an RFC 9457 Problem Details document."""
        return {
            "type": "about:blank",
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }


class CatalogSessionServer:
    """Minimal JSON-RPC server speaking over stdin/stdout."""

    def __init__(
        self,
        client: AgentCatalogClient,
        *,
        access: AccessController,
        audit: AuditLogger,
    ) -> None:
        self.client = client
        self._shutdown = False
        self._access = access
        self._audit = audit
        self._role = access.role.value
        self._methods: dict[str, Callable[[JsonObject], JsonValue]] = {
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
        }

    def _write(self, payload: JsonObject) -> None:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _respond(self, request_id: JsonValue, result: JsonValue) -> None:
        self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _dispatch(self, method: JsonValue, params: JsonObject) -> JsonValue:
        if not isinstance(method, str):
            raise CatalogSessionServerError(400, "invalid-request", "Missing method name")
        handler = self._methods.get(method)
        if handler is None:
            detail = f"Unknown method: {method}"
            raise CatalogSessionServerError(404, "unknown-method", detail)
        try:
            self._access.authorize(method)
        except PermissionError as exc:
            self._audit.log(action=method, role=self._role, status="forbidden", detail=str(exc))
            raise CatalogSessionServerError(403, "forbidden", str(exc)) from exc
        try:
            result = handler(params)
        except AgentCatalogClientError as exc:
            self._audit.log(action=method, role=self._role, status="catalog-error", detail=str(exc))
            raise CatalogSessionServerError(404, "catalog-error", str(exc)) from exc
        except (TypeError, ValueError) as exc:
            self._audit.log(
                action=method, role=self._role, status="invalid-params", detail=str(exc)
            )
            raise CatalogSessionServerError(400, "invalid-params", str(exc)) from exc
        self._audit.log(action=method, role=self._role, status="ok")
        return result

    def _error(self, request_id: JsonValue, error: CatalogSessionServerError) -> None:
        logger.debug("JSON-RPC error %s: %s", error.status, error.detail)
        self._write({"jsonrpc": "2.0", "id": request_id, "error": error.to_problem()})

    def serve(self) -> int:
        for raw_line in sys.stdin:
            message_line = raw_line.strip()
            if not message_line:
                continue
            try:
                message = json.loads(message_line)
            except json.JSONDecodeError as exc:
                error = CatalogSessionServerError(400, "invalid-json", str(exc))
                self._error(None, error)
                continue
            params = message.get("params")
            request_id = message.get("id")
            try:
                result = self._dispatch(message.get("method"), params or {})
            except CatalogSessionServerError as exc:
                self._error(request_id, exc)
                continue
            if request_id is not None:
                self._respond(request_id, result)
            if self._shutdown:
                break
        return 0

    def _handle_initialize(self, _: JsonObject) -> JsonObject:
        commands = sorted(name for name in self._methods if name.startswith("catalog."))
        return {"capabilities": {"procedures": commands}}

    def _handle_capabilities(self, _: JsonObject) -> list[str]:
        return [pkg.name for pkg in self.client.list_packages()]

    def _handle_symbol(self, params: JsonObject) -> JsonObject:
        symbol_id = str(params.get("symbol_id"))
        symbol = self.client.get_symbol(symbol_id)
        if symbol is None:
            detail = f"Unknown symbol: {symbol_id}"
            raise CatalogSessionServerError(404, "unknown-symbol", detail)
        return symbol.model_dump()

    def _handle_find_callers(self, params: JsonObject) -> list[str]:
        symbol_id = str(params.get("symbol_id"))
        return self.client.find_callers(symbol_id)

    def _handle_find_callees(self, params: JsonObject) -> list[str]:
        symbol_id = str(params.get("symbol_id"))
        return self.client.find_callees(symbol_id)

    def _handle_change_impact(self, params: JsonObject) -> JsonObject:
        symbol_id = str(params.get("symbol_id"))
        return self.client.change_impact(symbol_id).model_dump()

    def _handle_suggest_tests(self, params: JsonObject) -> list[JsonObject]:
        symbol_id = str(params.get("symbol_id"))
        return [dict(test) for test in self.client.suggest_tests(symbol_id)]

    def _handle_open_anchor(self, params: JsonObject) -> dict[str, str]:
        symbol_id = str(params.get("symbol_id"))
        return self.client.open_anchor(symbol_id)

    def _handle_search(self, params: JsonObject) -> list[JsonObject]:
        query = str(params.get("query", ""))
        k = int(params.get("k", 10))
        facets = params.get("facets")
        facet_map = {str(key): str(value) for key, value in (facets or {}).items()}
        results = self.client.search(query, k=k, facets=facet_map)
        return [asdict(result) for result in results]

    def _handle_list_modules(self, params: JsonObject) -> list[str]:
        package = str(params.get("package"))
        return [module.qualified for module in self.client.list_modules(package)]

    def _handle_shutdown(self, _: JsonObject) -> None:
        self._shutdown = True

    def _handle_exit(self, _: JsonObject) -> None:
        self._shutdown = True


def _resolve_role(value: str) -> Role:
    try:
        return Role(value)
    except ValueError as exc:
        valid = ", ".join(role.value for role in Role)
        raise ValueError(f"Role must be one of: {valid}") from exc


def main(argv: list[str] | None = None) -> int:
    """Run the stdio session server."""
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog_path = args.catalog
    if not catalog_path.exists():
        message = f"Catalog not found at {catalog_path}"
        raise SystemExit(message)
    repo_root = args.repo_root
    if not repo_root.exists():
        message = f"Repository root does not exist: {repo_root}"
        raise SystemExit(message)
    client = AgentCatalogClient.from_path(catalog_path, repo_root=repo_root)
    try:
        role = _resolve_role(args.role)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    access = AccessController(role=role, enabled=bool(args.hosted_mode))
    audit_logger = AuditLogger(args.audit_log, enabled=access.enabled)
    server = CatalogSessionServer(client, access=access, audit=audit_logger)
    return server.serve()


__all__ = ["CatalogSessionServer", "CatalogSessionServerError", "build_parser", "main"]
