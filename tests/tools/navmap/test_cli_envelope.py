"""Tests for shared CLI envelope helpers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import cast

import pytest
from msgspec import structs
from tools import (
    CliEnvelope,
    CliEnvelopeBuilder,
    CliStatus,
    build_problem_details,
    render_cli_envelope,
    validate_cli_envelope,
)

from kgfoundry_common.errors import SchemaValidationError


def _navmap_envelope(status: CliStatus) -> CliEnvelope:
    builder = CliEnvelopeBuilder.create(
        command="repair_navmaps", status=status, subcommand="repair"
    )
    return builder.finish(duration_seconds=0.0)


def test_cli_envelope_builder_minimal() -> None:
    envelope = _navmap_envelope("success")

    assert envelope.schemaVersion == "1.0.0"
    assert envelope.schemaId == "https://kgfoundry.dev/schema/cli-envelope.json"
    assert envelope.status == "success"
    assert envelope.command == "repair_navmaps"
    assert envelope.subcommand == "repair"
    assert envelope.durationSeconds == 0.0
    assert envelope.files == []
    assert envelope.errors == []


def test_cli_envelope_builder_sets_timestamp() -> None:
    envelope = _navmap_envelope("success")
    datetime.fromisoformat(envelope.generatedAt)


@pytest.mark.parametrize("status", ["success", "violation", "config", "error"])
def test_cli_envelope_builder_supports_statuses(status: CliStatus) -> None:
    envelope = _navmap_envelope(status)
    assert envelope.status == status


def test_validate_cli_envelope_accepts_valid_payload() -> None:
    envelope = structs.replace(_navmap_envelope("success"), durationSeconds=0.5)
    validate_cli_envelope(envelope)


def test_validate_cli_envelope_accepts_problem_details() -> None:
    problem = build_problem_details(
        type="https://kgfoundry.dev/problems/navmap-repair-error",
        title="Navmap repair failed",
        status=500,
        detail="File not found",
        instance="urn:navmap:repair:error",
    )
    envelope = structs.replace(_navmap_envelope("error"), problem=problem)
    validate_cli_envelope(envelope)


def test_validate_cli_envelope_rejects_invalid_status() -> None:
    envelope = structs.replace(
        _navmap_envelope("success"), status=cast(CliStatus, "invalid_status")
    )
    with pytest.raises(SchemaValidationError):
        validate_cli_envelope(envelope)


def test_validate_cli_envelope_rejects_invalid_schema_version() -> None:
    envelope = structs.replace(_navmap_envelope("success"), schemaVersion="0.0.0")
    with pytest.raises(SchemaValidationError):
        validate_cli_envelope(envelope)


def test_render_cli_envelope_outputs_valid_json() -> None:
    envelope = _navmap_envelope("success")
    json_str = render_cli_envelope(envelope)
    parsed = json.loads(json_str)
    assert parsed["status"] == "success"
    assert parsed["command"] == "repair_navmaps"


def test_render_cli_envelope_pretty_prints() -> None:
    envelope = _navmap_envelope("success")
    json_str = render_cli_envelope(envelope)
    assert "\n" in json_str
    assert json_str.count("\n") > 1


def test_render_cli_envelope_includes_problem_details() -> None:
    problem = build_problem_details(
        type="https://kgfoundry.dev/problems/navmap-repair-error",
        title="Navmap repair failed",
        status=500,
        detail="File not found",
        instance="urn:navmap:repair:error",
    )
    envelope = structs.replace(_navmap_envelope("error"), problem=problem)
    parsed = json.loads(render_cli_envelope(envelope))
    assert parsed["problem"]["status"] == 500
