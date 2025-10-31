"""Schema validation tests for the Agent Catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

SCHEMA_PATH = Path("docs/_build/schema_agent_catalog.json")
CATALOG_FIXTURE = Path("tests/fixtures/agent/catalog_sample.json")


def _load_validator() -> Draft202012Validator:
    """Return a draft 2020-12 validator for the catalog schema."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_catalog_fixture_validates_against_schema() -> None:
    """The sample catalog should validate cleanly against the schema."""
    validator = _load_validator()
    payload = json.loads(CATALOG_FIXTURE.read_text(encoding="utf-8"))
    validator.validate(payload)


def test_schema_rejects_invalid_remap_order() -> None:
    """Ordered sequences must remain arrays per the schema contract."""
    validator = _load_validator()
    payload = json.loads(CATALOG_FIXTURE.read_text(encoding="utf-8"))
    anchors = payload["packages"][0]["modules"][0]["symbols"][0]["anchors"]
    anchors["remap_order"] = {"symbol_id": "not-an-array"}
    with pytest.raises(ValidationError):
        validator.validate(payload)
