"""Audit logging helpers for hosted Agent Catalog deployments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append structured audit events to a JSONL file."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, action: str, role: str, status: str, detail: str | None = None) -> None:
        """Append an audit entry if logging is enabled."""
        if not self._enabled:
            return
        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "action": action,
            "role": role,
            "status": status,
        }
        if detail:
            payload["detail"] = detail
        with self._path.open("a", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.write("\n")


__all__ = ["AuditLogger"]
