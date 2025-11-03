"""Audit logging helpers for hosted Agent Catalog deployments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kgfoundry_common.logging import get_correlation_id

if TYPE_CHECKING:
    from pathlib import Path

    from kgfoundry_common.problem_details import JsonValue


class AuditLogger:
    """Append structured audit events to a JSONL file.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Describe ``path``.
    enabled : bool, optional
        Describe ``enabled``.
        Defaults to ``True``.
    """

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        """Document   init  .

        <!-- auto:docstring-builder v1 -->

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        path : Path
            Configure the path.
        enabled : bool, optional
            Indicate whether enabled. Defaults to ``True``.
            Defaults to ``True``.
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

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        action : str
            Describe ``action``.
        role : str
            Describe ``role``.
        status : str
            Describe ``status``.
        detail : str | NoneType, optional
            Describe ``detail``.
            Defaults to ``None``.
        correlation_id : str | NoneType, optional
            Describe ``correlation_id``.
            Defaults to ``None``.
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


__all__ = ["AuditLogger"]
