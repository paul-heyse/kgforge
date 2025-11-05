"""Audit logging helpers for hosted Agent Catalog deployments."""
# [nav:section public-api]

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kgfoundry_common.logging import get_correlation_id
from kgfoundry_common.navmap_loader import load_nav_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from kgfoundry_common.problem_details import JsonValue


# [nav:anchor AuditLogger]
class AuditLogger:
    """Append structured audit events to a JSONL file.

    Provides a simple audit logging mechanism that writes structured JSON
    events to a JSONL (JSON Lines) file. Each event includes timestamp, action,
    role, status, and optional detail and correlation_id fields.

    Parameters
    ----------
    path : Path
        Path to the JSONL audit log file. Parent directory will be created
        if it doesn't exist.
    enabled : bool, optional
        Whether audit logging is enabled. If False, log() calls are no-ops.
        Defaults to True.

    Attributes
    ----------
    _path : Path
        Path to the audit log file.
    _enabled : bool
        Whether audit logging is enabled.

    Examples
    --------
    >>> from pathlib import Path
    >>> logger = AuditLogger(Path("/tmp/audit.jsonl"))
    >>> logger.log(action="search", role="user", status="success")
    """

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        """Initialize the audit logger.

        Creates the audit logger with the specified log file path and enabled
        state. If enabled, creates the parent directory if it doesn't exist.

        Parameters
        ----------
        path : Path
            Path to the JSONL audit log file.
        enabled : bool, optional
            Whether audit logging is enabled. Defaults to True.
        """
        self._path = path
        self._enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        action: str,
        role: str,
        status: str,
        detail: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Append an audit entry if logging is enabled.

        Writes a structured JSON audit event to the log file. Each event
        includes timestamp, action, role, and status. Optionally includes
        detail and correlation_id fields if provided.

        Parameters
        ----------
        action : str
            Action being audited (e.g., "search", "open_anchor", "capabilities").
        role : str
            Role of the actor performing the action (e.g., "user", "admin").
        status : str
            Status of the action (e.g., "success", "error", "unauthorized").
        detail : str | None, optional
            Optional detailed information about the action. Defaults to None.
        correlation_id : str | None, optional
            Optional correlation ID for tracing requests. If None, attempts to
            retrieve from contextvars. Defaults to None.

        Notes
        -----
        If logging is disabled (_enabled=False), this method is a no-op.
        Events are appended to the JSONL file in ISO format with UTC timestamps.
        """
        if not self._enabled:
            return
        payload: dict[str, JsonValue] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "action": action,
            "role": role,
            "status": status,
        }
        if detail:
            payload["detail"] = detail
        correlation = correlation_id or get_correlation_id()
        if correlation:
            payload["correlation_id"] = correlation
        with self._path.open("a", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.write("\n")


__all__ = [
    "AuditLogger",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))
