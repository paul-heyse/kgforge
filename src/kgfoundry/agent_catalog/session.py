"""Client helpers for the ``catalogctl-mcp`` stdio session server.

The server reports errors using RFC 9457 Problem Details payloads. For example::

    {
        "type": "about:blank",
        "title": "unknown-method",
        "status": 404,
        "detail": "Unknown method: catalog.invalid",
    }

This module surfaces those errors via :class:`CatalogSessionError` while providing a
high-level API for invoking catalog operations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import cast

from kgfoundry_common.errors import CatalogSessionError
from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.problem_details import JsonValue

JsonObject = dict[str, JsonValue]

logger = get_logger(__name__)

# HTTP status code boundaries
_MIN_STATUS_CODE = 100
_MAX_STATUS_CODE = 599


class CatalogSession:
    """Maintain a JSON-RPC session with ``catalogctl-mcp``.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    command : Iterable[str] | None, optional
        Describe ``command``.
        Defaults to ``None``.
    catalog : Path | None, optional
        Describe ``catalog``.
        Defaults to ``None``.
    repo_root : Path | None, optional
        Describe ``repo_root``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        command: Iterable[str] | None = None,
        catalog: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        """Document   init  .

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        command : str | NoneType, optional
            Configure the command. Defaults to ``None``.
            Defaults to ``None``.
        catalog : Path | NoneType, optional
            Configure the catalog. Defaults to ``None``.
            Defaults to ``None``.
        repo_root : Path | NoneType, optional
            Configure the repo root. Defaults to ``None``.
            Defaults to ``None``.
        """
        command_sequence = command or self._default_command()
        self._command = [str(part) for part in command_sequence]
        if catalog is not None:
            self._command.extend(["--catalog", str(catalog)])
        if repo_root is not None:
            self._command.extend(["--repo-root", str(repo_root)])
        self._process: subprocess.Popen[str] | None = None
        self._lock = Lock()
        self._next_id = 1

    @staticmethod
    def _default_command() -> list[str]:
        """Return the default command used to spawn the stdio server.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        list[str]
            Describe return value.
        """
        return [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
        ]

    def _ensure_process(self) -> subprocess.Popen[str]:
        """Spawn the stdio server process if required.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        str
            Describe return value.
        """
        if self._process is None or self._process.poll() is not None:
            with with_fields(
                logger, operation="session_spawn", status="started", command=" ".join(self._command)
            ) as log_adapter:
                log_adapter.debug("Starting catalog session")
                try:
                    self._process = subprocess.Popen(  # noqa: S603 - trusted command
                        self._command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    log_adapter.info("Session spawned successfully", extra={"status": "success"})
                except OSError as exc:
                    log_adapter.exception("Failed to spawn session", exc_info=exc)
                    message = f"Unable to launch catalogctl-mcp: {exc}"
                    raise CatalogSessionError(
                        message,
                        cause=exc,
                        context={"command": " ".join(self._command)},
                    ) from exc
        return self._process

    def _write_payload(self, process: subprocess.Popen[str], payload: JsonObject) -> None:
        """Write a JSON payload to the process stdin.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        process : str
            Describe ``process``.
        payload : dict[str, object]
            Describe ``payload``.
        """
        stdin = process.stdin
        if stdin is None:
            message = "catalogctl-mcp stdin is unavailable"
            raise CatalogSessionError(message, context={"operation": "write_payload"})
        stdin.write(json.dumps(payload) + "\n")
        stdin.flush()

    def _read_response(self, process: subprocess.Popen[str]) -> JsonObject:
        """Read and decode a JSON-RPC response from stdout.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        process : str
            Describe ``process``.

        Returns
        -------
        dict[str, object]
            Describe return value.
        """
        stdout = process.stdout
        if stdout is None:
            message = "catalogctl-mcp stdout is unavailable"
            raise CatalogSessionError(message, context={"operation": "read_response"})
        line = stdout.readline()
        if not line:
            message = "catalogctl-mcp terminated unexpectedly"
            raise CatalogSessionError(message, context={"operation": "read_response"})
        try:
            parsed_raw: object = json.loads(line)
            if not isinstance(parsed_raw, dict):
                message = "Invalid JSON-RPC response: expected object"
                parsed_type_name = type(parsed_raw).__name__
                raise CatalogSessionError(
                    message,
                    context={"operation": "read_response", "parsed_type": parsed_type_name},
                )
            # isinstance check narrows type - mypy understands this
            parsed: dict[str, JsonValue] = parsed_raw
            return parsed
        except json.JSONDecodeError as exc:
            message = f"Received invalid JSON payload: {exc}"
            raise CatalogSessionError(
                message,
                cause=exc,
                context={"operation": "read_response"},
            ) from exc

    def _validate_jsonrpc_id(self, value: JsonValue) -> int | str:
        """Validate and return a JSON-RPC ID (must be string or number).

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        value : object
            Describe ``value``.

        Returns
        -------
        int | str
            Describe return value.
        """
        if isinstance(value, (str, int)):
            return value
        message = f"Invalid JSON-RPC ID: expected string or number, got {type(value).__name__}"
        raise CatalogSessionError(
            message,
            context={"id_value": str(value), "id_type": type(value).__name__},
        )

    def _validate_status_code(self, value: JsonValue) -> int:
        """Validate and return an HTTP status code.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        value : object
            Describe ``value``.

        Returns
        -------
        int
            Describe return value.
        """
        if isinstance(value, int):
            if _MIN_STATUS_CODE <= value <= _MAX_STATUS_CODE:
                return value
            message = (
                f"Invalid status code: {value} (must be {_MIN_STATUS_CODE}-{_MAX_STATUS_CODE})"
            )
            raise CatalogSessionError(message, context={"status": value})
        if isinstance(value, str):
            try:
                parsed = int(value)
                if _MIN_STATUS_CODE <= parsed <= _MAX_STATUS_CODE:
                    return parsed
                message = (
                    f"Invalid status code: {parsed} (must be {_MIN_STATUS_CODE}-{_MAX_STATUS_CODE})"
                )
                raise CatalogSessionError(message, context={"status": parsed, "original": value})
            except ValueError as exc:
                message = f"Invalid status code format: {value}"
                raise CatalogSessionError(message, cause=exc, context={"status": value}) from exc
        message = f"Invalid status code type: expected int or str, got {type(value).__name__}"
        raise CatalogSessionError(
            message, context={"status": value, "status_type": type(value).__name__}
        )

    def _send_request(self, method: str, params: JsonObject | None = None) -> JsonValue:
        """Send a JSON-RPC request and return the result payload.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        method : str
            Describe ``method``.
        params : dict[str, object] | NoneType, optional
            Describe ``params``.
            Defaults to ``None``.

        Returns
        -------
        object
            Describe return value.
        """
        with with_fields(
            logger, operation="jsonrpc_request", method=method, status="started"
        ) as log_adapter:
            process = self._ensure_process()
            with self._lock:
                request_id = self._next_id
                self._next_id += 1
            payload: JsonObject = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            self._write_payload(process, payload)
            response = self._read_response(process)
            # Validate JSON-RPC response structure
            if not isinstance(response.get("jsonrpc"), str) or response.get("jsonrpc") != "2.0":
                message = "Invalid JSON-RPC response: missing or invalid jsonrpc field"
                raise CatalogSessionError(
                    message,
                    context={"method": method, "response_keys": list(response.keys())},
                )
            response_id = response.get("id")
            if response_id is not None:
                validated_id = self._validate_jsonrpc_id(response_id)
                if validated_id != request_id:
                    message = f"JSON-RPC ID mismatch: expected {request_id}, got {validated_id}"
                    raise CatalogSessionError(
                        message,
                        context={
                            "method": method,
                            "expected_id": request_id,
                            "received_id": validated_id,
                        },
                    )
            error = response.get("error")
            if isinstance(error, dict):
                error_type = str(error.get("type", "about:blank"))
                error_title = str(error.get("title", "unknown"))
                error_status = self._validate_status_code(error.get("status", 500))
                error_detail = str(error.get("detail", ""))
                log_adapter.error(
                    "JSON-RPC error response",
                    extra={
                        "status": "error",
                        "error_type": error_type,
                        "error_title": error_title,
                        "error_status": error_status,
                    },
                )
                raise CatalogSessionError(
                    error_detail or error_title,
                    context={
                        "method": method,
                        "error_type": error_type,
                        "error_title": error_title,
                        "error_status": error_status,
                    },
                )
            result = response.get("result")
            log_adapter.info("JSON-RPC request completed", extra={"status": "success"})
            return result

    def initialize(self) -> JsonObject:
        """Perform the initial handshake and return advertised capabilities.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        dict[str, object]
            Describe return value.
        """
        result = self._send_request("initialize")
        # result is JsonValue, narrow to JsonObject (dict)
        if isinstance(result, dict):
            return result  # isinstance narrows to dict[str, JsonValue] which is JsonObject
        return {}

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        facets: dict[str, str] | None = None,
    ) -> list[JsonObject]:
        """Execute hybrid search and return scored results.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int, optional
            Describe ``k``.
            Defaults to ``10``.
        facets : dict[str, str] | NoneType, optional
            Describe ``facets``.
            Defaults to ``None``.

        Returns
        -------
        list[dict[str, object]]
            Describe return value.
        """
        # Build params dict - facets dict[str, str] is compatible with JsonValue
        facets_dict: dict[str, str] = facets or {}
        params: JsonObject = {
            "query": query,
            "k": k,
            "facets": cast(JsonValue, facets_dict),  # dict[str, str] is JsonValue-compatible
        }
        result = self._send_request("catalog.search", params)
        if isinstance(result, list):
            return [cast(JsonObject, entry) for entry in result]
        return []

    def symbol(self, symbol_id: str) -> JsonObject:
        """Return catalog metadata for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        dict[str, object]
            Describe return value.
        """
        result = self._send_request("catalog.symbol", {"symbol_id": symbol_id})
        # result is JsonValue, narrow to JsonObject (dict)
        if isinstance(result, dict):
            return result  # isinstance narrows to dict[str, JsonValue] which is JsonObject
        return {}

    def find_callers(self, symbol_id: str) -> list[str]:
        """Return callers recorded for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        list[str]
            Describe return value.
        """
        result = self._send_request("catalog.find_callers", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def find_callees(self, symbol_id: str) -> list[str]:
        """Return callees recorded for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        list[str]
            Describe return value.
        """
        result = self._send_request("catalog.find_callees", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def change_impact(self, symbol_id: str) -> JsonObject:
        """Return change impact metadata for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        dict[str, object]
            Describe return value.
        """
        result = self._send_request("catalog.change_impact", {"symbol_id": symbol_id})
        # result is JsonValue, narrow to JsonObject (dict)
        if isinstance(result, dict):
            return result  # isinstance narrows to dict[str, JsonValue] which is JsonObject
        return {}

    def suggest_tests(self, symbol_id: str) -> list[JsonObject]:
        """Return suggested test metadata for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        list[dict[str, object]]
            Describe return value.
        """
        result = self._send_request("catalog.suggest_tests", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [cast(JsonObject, entry) for entry in result]
        return []

    def open_anchor(self, symbol_id: str) -> dict[str, str]:
        """Return editor and GitHub anchors for ``symbol_id``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        symbol_id : str
            Describe ``symbol_id``.

        Returns
        -------
        dict[str, str]
            Describe return value.
        """
        result = self._send_request("catalog.open_anchor", {"symbol_id": symbol_id})
        if isinstance(result, dict):
            return {str(key): str(value) for key, value in result.items()}
        return {}

    def list_modules(self, package: str) -> list[str]:
        """Return module names for ``package``.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        package : str
            Describe ``package``.

        Returns
        -------
        list[str]
            Describe return value.
        """
        result = self._send_request("catalog.list_modules", {"package": package})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def shutdown(self) -> None:
        """Request a graceful shutdown of the underlying process.

        <!-- auto:docstring-builder v1 -->
        """
        with with_fields(logger, operation="session_shutdown", status="started") as log_adapter:
            try:
                self._send_request("session.shutdown")
                log_adapter.info("Shutdown request successful", extra={"status": "success"})
            except CatalogSessionError as exc:
                log_adapter.debug(
                    "Shutdown request failed", extra={"status": "warning", "error": str(exc)}
                )
            finally:
                self.close()

    def close(self) -> None:
        """Terminate the session process if it is running.

        <!-- auto:docstring-builder v1 -->
        """
        process = self._process
        if process is None:
            return
        stdin = process.stdin
        if stdin is not None:
            stdin.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive guard
            process.kill()
        self._process = None

    def __enter__(self) -> CatalogSession:
        """Start the session process and return ``self``.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        CatalogSession
            Describe return value.
        """
        self._ensure_process()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Ensure the underlying process is terminated when leaving the context.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        exc_type : BaseException | NoneType
            Describe ``exc_type``.
        exc : BaseException | NoneType
            Describe ``exc``.
        tb : traceback | NoneType
            Describe ``tb``.
        """
        _ = exc_type, exc, tb
        self.close()


__all__ = ["CatalogSession", "CatalogSessionError", "JsonObject", "JsonValue"]
