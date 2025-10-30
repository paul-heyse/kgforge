"""Client helpers for the ``catalogctl-mcp`` stdio session server.

The server reports errors using RFC 9457 Problem Details payloads. For example::

    {
        "type": "about:blank",
        "title": "unknown-method",
        "status": 404,
        "detail": "Unknown method: catalog.invalid"
    }

This module surfaces those errors via :class:`CatalogSessionError` while providing a
high-level API for invoking catalog operations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import cast

JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
JsonObject = dict[str, JsonValue]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(slots=True)
class ProblemDetails:
    """RFC 9457 error payload returned by the session server."""

    type: str
    title: str
    status: int
    detail: str

    def as_dict(self) -> JsonObject:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "type": self.type,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }


class CatalogSessionError(RuntimeError):
    """Raised when the stdio session reports an error."""

    def __init__(self, problem: ProblemDetails) -> None:
        super().__init__(problem.detail)
        self.problem = problem


class CatalogSession:
    """Maintain a JSON-RPC session with ``catalogctl-mcp``."""

    def __init__(
        self,
        *,
        command: Iterable[str] | None = None,
        catalog: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
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
        """Return the default command used to spawn the stdio server."""
        return [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
        ]

    def _ensure_process(self) -> subprocess.Popen[str]:
        """Spawn the stdio server process if required."""
        if self._process is None or self._process.poll() is not None:
            logger.debug("Starting catalog session: %s", self._command)
            try:
                self._process = subprocess.Popen(  # noqa: S603 - trusted command
                    self._command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except OSError as exc:
                problem = ProblemDetails(
                    type="about:blank",
                    title="spawn-failed",
                    status=500,
                    detail=f"Unable to launch catalogctl-mcp: {exc}",
                )
                raise CatalogSessionError(problem) from exc
        return self._process

    def _write_payload(self, process: subprocess.Popen[str], payload: JsonObject) -> None:
        """Write a JSON payload to the process stdin."""
        stdin = process.stdin
        if stdin is None:
            problem = ProblemDetails(
                type="about:blank",
                title="stdin-closed",
                status=500,
                detail="catalogctl-mcp stdin is unavailable",
            )
            raise CatalogSessionError(problem)
        stdin.write(json.dumps(payload) + "\n")
        stdin.flush()

    def _read_response(self, process: subprocess.Popen[str]) -> JsonObject:
        """Read and decode a JSON-RPC response from stdout."""
        stdout = process.stdout
        if stdout is None:
            problem = ProblemDetails(
                type="about:blank",
                title="stdout-closed",
                status=500,
                detail="catalogctl-mcp stdout is unavailable",
            )
            raise CatalogSessionError(problem)
        line = stdout.readline()
        if not line:
            problem = ProblemDetails(
                type="about:blank",
                title="session-closed",
                status=500,
                detail="catalogctl-mcp terminated unexpectedly",
            )
            raise CatalogSessionError(problem)
        try:
            return cast(JsonObject, json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            problem = ProblemDetails(
                type="about:blank",
                title="invalid-json",
                status=500,
                detail=f"Received invalid JSON payload: {exc}",
            )
            raise CatalogSessionError(problem) from exc

    def _send_request(self, method: str, params: JsonObject | None = None) -> JsonValue:
        """Send a JSON-RPC request and return the result payload."""
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
        error = response.get("error")
        if isinstance(error, dict):
            problem = ProblemDetails(
                type=str(error.get("type", "about:blank")),
                title=str(error.get("title", "unknown")),
                status=int(error.get("status", 500)),
                detail=str(error.get("detail", "")),
            )
            raise CatalogSessionError(problem)
        return response.get("result")

    def initialize(self) -> JsonObject:
        """Perform the initial handshake and return advertised capabilities."""
        result = self._send_request("initialize")
        return cast(JsonObject, result or {})

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        facets: dict[str, str] | None = None,
    ) -> list[JsonObject]:
        """Execute hybrid search and return scored results."""
        result = self._send_request(
            "catalog.search",
            {"query": query, "k": k, "facets": facets or {}},
        )
        if isinstance(result, list):
            return [cast(JsonObject, entry) for entry in result]
        return []

    def symbol(self, symbol_id: str) -> JsonObject:
        """Return catalog metadata for ``symbol_id``."""
        result = self._send_request("catalog.symbol", {"symbol_id": symbol_id})
        return cast(JsonObject, result or {})

    def find_callers(self, symbol_id: str) -> list[str]:
        """Return callers recorded for ``symbol_id``."""
        result = self._send_request("catalog.find_callers", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def find_callees(self, symbol_id: str) -> list[str]:
        """Return callees recorded for ``symbol_id``."""
        result = self._send_request("catalog.find_callees", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def change_impact(self, symbol_id: str) -> JsonObject:
        """Return change impact metadata for ``symbol_id``."""
        result = self._send_request("catalog.change_impact", {"symbol_id": symbol_id})
        return cast(JsonObject, result or {})

    def suggest_tests(self, symbol_id: str) -> list[JsonObject]:
        """Return suggested test metadata for ``symbol_id``."""
        result = self._send_request("catalog.suggest_tests", {"symbol_id": symbol_id})
        if isinstance(result, list):
            return [cast(JsonObject, entry) for entry in result]
        return []

    def open_anchor(self, symbol_id: str) -> dict[str, str]:
        """Return editor and GitHub anchors for ``symbol_id``."""
        result = self._send_request("catalog.open_anchor", {"symbol_id": symbol_id})
        if isinstance(result, dict):
            return {str(key): str(value) for key, value in result.items()}
        return {}

    def list_modules(self, package: str) -> list[str]:
        """Return module names for ``package``."""
        result = self._send_request("catalog.list_modules", {"package": package})
        if isinstance(result, list):
            return [str(entry) for entry in result]
        return []

    def shutdown(self) -> None:
        """Request a graceful shutdown of the underlying process."""
        try:
            self._send_request("session.shutdown")
        except CatalogSessionError as exc:  # pragma: no cover - defensive guard
            logger.debug("Shutdown request failed: %s", exc.problem.as_dict())
        finally:
            self.close()

    def close(self) -> None:
        """Terminate the session process if it is running."""
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
        """Start the session process and return ``self``."""
        self._ensure_process()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Ensure the underlying process is terminated when leaving the context."""
        _ = exc_type, exc, tb
        self.close()


__all__ = ["CatalogSession", "CatalogSessionError", "ProblemDetails"]
