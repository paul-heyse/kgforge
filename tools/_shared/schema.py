"""Shared schema helpers for tooling payload validation.

This module centralises JSON Schema lookups for tooling artefacts. It exposes
helpers for resolving schema paths under ``schema/tools`` and validating payloads
using the canonical utilities in :mod:`kgfoundry_common.serialization`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from kgfoundry_common.serialization import validate_payload

SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schema" / "tools"

__all__ = ["get_schema_path", "validate_tools_payload"]

_SCHEMA_CACHE: dict[str, Path] = {}


def get_schema_path(name: str) -> Path:
    """Return the absolute path for a tooling schema.

    Parameters
    ----------
    name : str
        Basename of the schema file under ``schema/tools``.

    Returns
    -------
    Path
        Absolute path to the schema file.

    Raises
    ------
    FileNotFoundError
        If the schema file is missing.
    """
    candidate = SCHEMA_ROOT / name
    if not candidate.exists():
        msg = f"Tooling schema not found: {candidate}"
        raise FileNotFoundError(msg)
    cached = _SCHEMA_CACHE.get(name)
    if cached is not None:
        return cached
    _SCHEMA_CACHE[name] = candidate
    return candidate


def validate_tools_payload(payload: Mapping[str, object], schema_name: str) -> None:
    """Validate ``payload`` against a tooling schema.

    Parameters
    ----------
    payload : Mapping[str, object]
        JSON-serialisable payload to validate.
    schema_name : str
        Basename of the schema under ``schema/tools``.
    """
    schema_path = get_schema_path(schema_name)
    validate_payload(payload, schema_path)
