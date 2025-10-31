"""Tests for the agent API OpenAPI document generator."""

from __future__ import annotations

import json
from pathlib import Path

from tools.docs import build_agent_api


def test_build_spec_contains_required_components() -> None:
    """The generated specification should expose ProblemDetails and Symbol schemas."""
    spec = build_agent_api.build_spec()
    assert spec["openapi"] == "3.2.0"
    components = spec["components"]["schemas"]
    assert "ProblemDetails" in components
    assert "Symbol" in components
    remap = components["Symbol"]["properties"]["anchors"]["properties"]["remap_order"]
    assert remap["type"] == "array"


def test_main_writes_openapi_document(tmp_path: Path) -> None:
    """Running ``main`` should write the OpenAPI document to disk."""
    target = tmp_path / "agent_api_openapi.json"
    original = build_agent_api.OUTPUT
    build_agent_api.OUTPUT = target
    try:
        assert build_agent_api.main() == 0
    finally:
        build_agent_api.OUTPUT = original
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["paths"], "expected documented endpoints"
