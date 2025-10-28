"""Validate that exported JSON schema files are syntactically correct."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema


def test_all_jsonschemas_are_valid() -> None:
    schema_dir = Path("docs/reference/schemas")
    files = sorted(schema_dir.glob("*.json"))
    assert files, "no schema files generated; run tools/docs/export_schemas.py"
    for path in files:
        data = json.loads(path.read_text())
        jsonschema.Draft202012Validator.check_schema(data)
