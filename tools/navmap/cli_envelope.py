"""Build CLI envelope payload for navmap repair operations.

This module provides helpers for constructing the base CLI envelope schema
conforming to schema/tools/cli_envelope.json for navmap repair operations.

Examples
--------
>>> from tools.navmap.cli_envelope import (
...     build_cli_envelope_skeleton,
...     validate_cli_envelope,
...     render_cli_envelope,
... )
>>> from tools._shared.problem_details import build_problem_details
>>> envelope = build_cli_envelope_skeleton("success")
>>> envelope["durationSeconds"] = 0.5
>>> envelope["command"] = "repair_navmaps"
>>> validate_cli_envelope(envelope)
>>> json_str = render_cli_envelope(envelope)
>>> assert "success" in json_str

>>> # Error case with Problem Details
>>> error_envelope = build_cli_envelope_skeleton("error")
>>> error_envelope["problem"] = build_problem_details(
...     type="https://kgfoundry.dev/problems/navmap-repair-error",
...     title="Navmap repair failed",
...     status=500,
...     detail="File not found",
...     instance="urn:navmap:repair:error",
... )
>>> validate_cli_envelope(error_envelope)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

CLI_ENVELOPE_SCHEMA_VERSION = "1.0.0"
CLI_ENVELOPE_SCHEMA_ID = "https://kgfoundry.dev/schema/cli-envelope.json"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_ENVELOPE_SCHEMA_PATH = _REPO_ROOT / "schema" / "tools" / "cli_envelope.json"

_cli_envelope_validator: Draft202012Validator | None = None


def _load_cli_envelope_validator() -> Draft202012Validator:
    """Load and cache the CLI envelope schema validator."""
    global _cli_envelope_validator  # noqa: PLW0603
    if _cli_envelope_validator is None:
        with _CLI_ENVELOPE_SCHEMA_PATH.open(encoding="utf-8") as f:
            schema = json.load(f)
        _cli_envelope_validator = Draft202012Validator(schema)
    return _cli_envelope_validator


def build_cli_envelope_skeleton(status: str) -> dict[str, Any]:
    """Build a minimal CLI envelope payload with required fields.

    Parameters
    ----------
    status : str
        Run status ("success", "violation", "config", or "error").

    Returns
    -------
    dict[str, Any]
        CLI envelope payload skeleton.
    """
    return {
        "schemaVersion": CLI_ENVELOPE_SCHEMA_VERSION,
        "schemaId": CLI_ENVELOPE_SCHEMA_ID,
        "generatedAt": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "status": status,
        "command": "repair_navmaps",
        "subcommand": "",
        "durationSeconds": 0.0,
        "files": [],
        "errors": [],
    }


def validate_cli_envelope(payload: dict[str, Any]) -> None:
    """Validate CLI envelope payload against the schema.

    Parameters
    ----------
    payload : dict[str, Any]
        CLI envelope payload to validate.

    Raises
    ------
    ValueError
        If validation fails.
    """
    validator = _load_cli_envelope_validator()
    errors = list(validator.iter_errors(payload))
    if errors:
        error_messages = [str(err) for err in errors]
        error_text = ", ".join(error_messages)
        msg = f"CLI envelope validation failed: {error_text}"
        raise ValueError(msg)


def render_cli_envelope(payload: dict[str, Any]) -> str:
    """Render CLI envelope as JSON string.

    Parameters
    ----------
    payload : dict[str, Any]
        CLI envelope payload.

    Returns
    -------
    str
        JSON-encoded CLI envelope (pretty-printed).
    """
    return json.dumps(payload, indent=2)
