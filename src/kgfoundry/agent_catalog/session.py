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
import sys
from threading import Lock
from typing import TYPE_CHECKING, Self, cast

from kgfoundry_common.errors import CatalogSessionError
from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.problem_details import JsonValue
from kgfoundry_common.subprocess_utils import TimeoutExpired, spawn_text_process

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from types import TracebackType

    from kgfoundry_common.subprocess_utils import TextProcess

JsonObject = dict[str, JsonValue]

logger = get_logger(__name__)

# HTTP status code boundaries
_MIN_STATUS_CODE = 100
_MAX_STATUS_CODE = 599


class CatalogSession:
    """Maintain a JSON-RPC session with ``catalogctl-mcp``.

    Provides a client interface for communicating with the catalogctl-mcp
    stdio server via JSON-RPC. Handles process lifecycle, request/response
    serialization, and error handling with RFC 9457 Problem Details.

    Parameters
    ----------
    command : Iterable[str] | None, optional
        Command sequence to spawn the stdio server. If None, uses default.
        Defaults to None.
    catalog : Path | None, optional
        Path to catalog JSON file. Defaults to None.
    repo_root : Path | None, optional
        Repository root for resolving anchors. Defaults to None.
    """

    def __init__(
        self,
        *,
        command: Iterable[str] | None = None,
        catalog: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        """Initialize catalog session.

        Sets up command arguments and initializes session state. The process
        is spawned lazily on first request.

        Parameters
        ----------
        command : Iterable[str] | None, optional
            Command sequence to spawn server. If None, uses default.
            Defaults to None.
        catalog : Path | None, optional
            Path to catalog JSON file. Defaults to None.
        repo_root : Path | None, optional
            Repository root for resolving anchors. Defaults to None.
        """
        command_sequence = command or self._default_command()
        self._command = [str(part) for part in command_sequence]
        if catalog is not None:
            self._command.extend(["--catalog", str(catalog)])
        if repo_root is not None:
            self._command.extend(["--repo-root", str(repo_root)])
        self._process: TextProcess | None = None
        self._lock = Lock()
        self._next_id = 1

    @staticmethod
    def _default_command() -> list[str]:
        """Return the default command used to spawn the stdio server.

        Returns the standard command sequence for launching catalogctl-mcp
        using the current Python interpreter.

        Returns
        -------
        list[str]
            Command sequence: [sys.executable, "-m", "tools.agent_catalog.catalogctl_mcp"].
        """
        return [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
        ]

    def _ensure_process(self) -> TextProcess:
        """Spawn the stdio server process if required.

        Lazily spawns the catalogctl-mcp process if it doesn't exist or
        has terminated. Thread-safe.

        Returns
        -------
        TextProcess
            Active text process for the stdio server.

        Raises
        ------
        CatalogSessionError
            If the process cannot be spawned.
        """
        if self._process is None or self._process.poll() is not None:
            with with_fields(
                logger, operation="session_spawn", status="started", command=" ".join(self._command)
            ) as log_adapter:
                log_adapter.debug("Starting catalog session")
                try:
                    self._process = spawn_text_process(self._command)
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

    @staticmethod
    def _write_payload(process: TextProcess, payload: JsonObject) -> None:
        """Write a JSON payload to the process stdin.

        Serializes the payload to JSON and writes it as a newline-terminated
        line to the process stdin.

        Parameters
        ----------
        process : TextProcess
            Text process with stdin available.
        payload : JsonObject
            JSON-RPC payload dictionary to write.

        Raises
        ------
        CatalogSessionError
            If process stdin is unavailable.
        """
        stdin = process.stdin
        if stdin is None:
            message = "catalogctl-mcp stdin is unavailable"
            raise CatalogSessionError(message, context={"operation": "write_payload"})
        stdin.write(json.dumps(payload) + "\n")
        stdin.flush()

    @staticmethod
    def _read_response(process: TextProcess) -> JsonObject:
        """Read and decode a JSON-RPC response from stdout.

        Reads a single line from stdout and parses it as JSON. Validates
        the response is a JSON object.

        Parameters
        ----------
        process : TextProcess
            Text process with stdout available.

        Returns
        -------
        JsonObject
            Parsed JSON-RPC response dictionary.

        Raises
        ------
        CatalogSessionError
            If process stdout is unavailable, process terminates unexpectedly,
            or response is invalid JSON.
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
        except json.JSONDecodeError as exc:
            message = f"Received invalid JSON payload: {exc}"
            raise CatalogSessionError(
                message,
                cause=exc,
                context={"operation": "read_response"},
            ) from exc
        else:
            if not isinstance(parsed_raw, dict):
                message = "Invalid JSON-RPC response: expected object"
                parsed_type_name = type(parsed_raw).__name__
                raise CatalogSessionError(
                    message,
                    context={"operation": "read_response", "parsed_type": parsed_type_name},
                )
            parsed: dict[str, JsonValue] = parsed_raw
            return parsed

    @staticmethod
    def _validate_jsonrpc_id(value: JsonValue) -> int | str:
        """Validate and return a JSON-RPC ID (must be string or number).

        Ensures the ID conforms to JSON-RPC 2.0 specification which requires
        IDs to be strings or numbers.

        Parameters
        ----------
        value : JsonValue
            ID value to validate.

        Returns
        -------
        int | str
            Validated ID value.

        Raises
        ------
        CatalogSessionError
            If value is not a string or integer.
        """
        if isinstance(value, (str, int)):
            return value
        message = f"Invalid JSON-RPC ID: expected string or number, got {type(value).__name__}"
        raise CatalogSessionError(
            message,
            context={"id_value": str(value), "id_type": type(value).__name__},
        )

    @staticmethod
    def _validate_status_code(value: JsonValue) -> int:
        """Validate and return an HTTP status code.

        Validates that the status code is an integer in the valid HTTP range
        (100-599) or a parseable string representation.

        Parameters
        ----------
        value : JsonValue
            Status code value to validate (int or str).

        Returns
        -------
        int
            Validated status code integer.

        Raises
        ------
        CatalogSessionError
            If value is not a valid status code (integer in valid range or
            parseable string).
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

        Constructs a JSON-RPC 2.0 request, sends it to the server, and
        validates the response. Handles error responses by raising
        CatalogSessionError with Problem Details.

        Parameters
        ----------
        method : str
            JSON-RPC method name (e.g., "catalog.search").
        params : JsonObject | None, optional
            Method parameters dictionary. Defaults to None (empty dict).

        Returns
        -------
        JsonValue
            Result payload from successful response.

        Raises
        ------
        CatalogSessionError
            If the request fails, response is invalid, or JSON-RPC error
            is returned.
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

        Sends the "initialize" JSON-RPC method to establish the session
        and retrieve server capabilities.

        Returns
        -------
        JsonObject
            Capabilities dictionary from server response.
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

        Performs semantic search over the catalog with optional facet filters.

        Parameters
        ----------
        query : str
            Search query text.
        k : int, optional
            Number of results to return. Defaults to 10.
        facets : dict[str, str] | None, optional
            Facet filters (package, module, kind, stability). Defaults to None.

        Returns
        -------
        list[JsonObject]
            List of search result dictionaries with scores and metadata.
        """
        # Build params dict - facets dict[str, str] is compatible with JsonValue
        facets_dict: dict[str, str] = facets or {}
        params: JsonObject = {
            "query": query,
            "k": k,
            "facets": cast("JsonValue", facets_dict),  # dict[str, str] is JsonValue-compatible
        }
        result = self._send_request("catalog.search", params)
        if isinstance(result, list):
            return [cast("JsonObject", entry) for entry in result]
        return []

    def symbol(self, symbol_id: str) -> JsonObject:
        """Return catalog metadata for ``symbol_id``.

        Retrieves symbol metadata including docfacts, anchors, and metrics.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier (e.g., "py:kgfoundry.search.find_similar").

        Returns
        -------
        JsonObject
            Symbol metadata dictionary.
        """
        result = self._send_request("catalog.symbol", {"symbol_id": symbol_id})
        # result is JsonValue, narrow to JsonObject (dict)
        if isinstance(result, dict):
            return result  # isinstance narrows to dict[str, JsonValue] which is JsonObject
        return {}

    def find_callers(self, symbol_id: str) -> list[str]:
        """Return callers recorded for ``symbol_id``.

        Finds all symbols that reference the given symbol.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        list[str]
            List of caller symbol IDs.
        """
        result = self._send_request("catalog.find_callers", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def find_callees(self, symbol_id: str) -> list[str]:
        """Return callees recorded for ``symbol_id``.

        Finds all symbols that are called by the given symbol.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        list[str]
            List of callee symbol IDs.
        """
        result = self._send_request("catalog.find_callees", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def change_impact(self, symbol_id: str) -> JsonObject:
        """Return change impact metadata for ``symbol_id``.

        Analyzes which symbols and modules would be affected by changes
        to the given symbol.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        JsonObject
            Change impact dictionary with affected symbols and modules.
        """
        result = self._send_request("catalog.change_impact", {"symbol_id": symbol_id})
        # result is JsonValue, narrow to JsonObject (dict)
        if isinstance(result, dict):
            return result  # isinstance narrows to dict[str, JsonValue] which is JsonObject
        return {}

    def suggest_tests(self, symbol_id: str) -> list[JsonObject]:
        """Return suggested test metadata for ``symbol_id``.

        Retrieves test suggestions including test files and test functions
        that should be run when the symbol changes.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        list[JsonObject]
            List of test suggestion dictionaries.
        """
        result = self._send_request("catalog.suggest_tests", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [cast("JsonObject", entry) for entry in result]
        return []

    def open_anchor(self, symbol_id: str) -> dict[str, str]:
        """Return editor and GitHub anchors for ``symbol_id``.

        Returns anchor information suitable for opening the symbol in
        an editor or viewing it on GitHub.

        Parameters
        ----------
        symbol_id : str
            Unique symbol identifier.

        Returns
        -------
        dict[str, str]
            Anchor dictionary with file path and line number keys.
        """
        result = self._send_request("catalog.open_anchor", {"symbol_id": symbol_id})
        if isinstance(result, dict):
            return {str(key): str(value) for key, value in result.items()}
        return {}

    def list_modules(self, package: str) -> list[str]:
        """Return module names for ``package``.

        Lists all modules contained in the specified package.

        Parameters
        ----------
        package : str
            Package name.

        Returns
        -------
        list[str]
            List of qualified module names.
        """
        result = self._send_request("catalog.list_modules", {"package": package})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def shutdown(self) -> None:
        """Request a graceful shutdown of the underlying process.

        Sends a shutdown request to the server and then closes the session. Logs shutdown status but
        does not raise exceptions.
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

        Closes stdin, terminates the process, and waits for it to exit. Force-kills if termination
        times out.
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
        except TimeoutExpired:  # pragma: no cover - defensive guard
            process.kill()
        self._process = None

    def __enter__(self) -> Self:
        """Start the session process and return ``self``.

        Context manager entry point. Spawns the process if needed.

        Returns
        -------
        CatalogSession
            Self instance for use in context manager.
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

        Context manager exit point. Always closes the session regardless
        of exceptions.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type (unused).
        exc : BaseException | None
            Exception instance (unused).
        tb : TracebackType | None
            Traceback object (unused).
        """
        _ = exc_type, exc, tb
        self.close()


__all__ = ["CatalogSession", "CatalogSessionError", "JsonObject", "JsonValue"]
