"""Tests for tools.navmap.cli_envelope module."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from tools._shared.problem_details import build_problem_details
from tools.navmap.cli_envelope import (
    build_cli_envelope_skeleton,
    render_cli_envelope,
    validate_cli_envelope,
)


class TestBuildCliEnvelopeSkeleton:
    """Tests for build_cli_envelope_skeleton function."""

    def test_builds_minimal_envelope(self) -> None:
        """build_cli_envelope_skeleton creates valid envelope."""
        envelope = build_cli_envelope_skeleton("success")

        assert envelope["schemaVersion"] == "1.0.0"
        assert envelope["schemaId"] == "https://kgfoundry.dev/schema/cli-envelope.json"
        assert envelope["status"] == "success"
        assert envelope["command"] == "repair_navmaps"
        assert envelope["subcommand"] == ""
        assert envelope["durationSeconds"] == 0.0
        assert envelope["files"] == []
        assert envelope["errors"] == []

    def test_includes_timestamp(self) -> None:
        """build_cli_envelope_skeleton includes generatedAt timestamp."""
        envelope = build_cli_envelope_skeleton("success")

        assert "generatedAt" in envelope
        # Should be valid ISO 8601 timestamp
        datetime.fromisoformat(envelope["generatedAt"])

    @pytest.mark.parametrize("status", ["success", "violation", "config", "error"])
    def test_supports_all_statuses(self, status: str) -> None:
        """build_cli_envelope_skeleton supports all valid statuses."""
        envelope = build_cli_envelope_skeleton(status)
        assert envelope["status"] == status


class TestValidateCliEnvelope:
    """Tests for validate_cli_envelope function."""

    def test_validates_success_envelope(self) -> None:
        """validate_cli_envelope accepts valid success envelope."""
        envelope = build_cli_envelope_skeleton("success")
        envelope["durationSeconds"] = 0.5
        validate_cli_envelope(envelope)  # Should not raise

    def test_validates_error_envelope_with_problem_details(self) -> None:
        """validate_cli_envelope accepts envelope with Problem Details."""
        envelope = build_cli_envelope_skeleton("error")
        envelope["problem"] = build_problem_details(
            type="https://kgfoundry.dev/problems/navmap-repair-error",
            title="Navmap repair failed",
            status=500,
            detail="File not found",
            instance="urn:navmap:repair:error",
        )
        validate_cli_envelope(envelope)  # Should not raise

    def test_validates_envelope_with_files(self) -> None:
        """validate_cli_envelope accepts envelope with file entries."""
        envelope = build_cli_envelope_skeleton("violation")
        envelope["files"] = [
            {"path": "site/_build/navmap/navmap.json", "status": "violation", "message": "Invalid"}
        ]
        validate_cli_envelope(envelope)  # Should not raise

    def test_validates_envelope_with_errors(self) -> None:
        """validate_cli_envelope accepts envelope with error entries."""
        envelope = build_cli_envelope_skeleton("error")
        envelope["errors"] = [{"status": "error", "message": "Test error"}]
        validate_cli_envelope(envelope)  # Should not raise

    def test_rejects_invalid_status(self) -> None:
        """validate_cli_envelope rejects envelope with invalid status."""
        envelope = build_cli_envelope_skeleton("success")
        envelope["status"] = "invalid_status"
        with pytest.raises(ValueError, match="validation failed"):
            validate_cli_envelope(envelope)

    def test_rejects_missing_required_field(self) -> None:
        """validate_cli_envelope rejects envelope missing required fields."""
        envelope = build_cli_envelope_skeleton("success")
        del envelope["schemaVersion"]
        with pytest.raises(ValueError, match="validation failed"):
            validate_cli_envelope(envelope)


class TestRenderCliEnvelope:
    """Tests for render_cli_envelope function."""

    def test_renders_valid_json(self) -> None:
        """render_cli_envelope produces valid JSON."""
        envelope = build_cli_envelope_skeleton("success")
        json_str = render_cli_envelope(envelope)

        parsed = json.loads(json_str)
        assert parsed["status"] == "success"
        assert parsed["command"] == "repair_navmaps"

    def test_renders_pretty_printed(self) -> None:
        """render_cli_envelope produces pretty-printed JSON."""
        envelope = build_cli_envelope_skeleton("success")
        json_str = render_cli_envelope(envelope)

        # Pretty-printed JSON should have newlines
        assert "\n" in json_str
        assert json_str.count("\n") > 1

    def test_renders_envelope_with_problem_details(self) -> None:
        """render_cli_envelope includes Problem Details in JSON."""
        envelope = build_cli_envelope_skeleton("error")
        envelope["problem"] = build_problem_details(
            type="https://kgfoundry.dev/problems/navmap-repair-error",
            title="Navmap repair failed",
            status=500,
            detail="File not found",
            instance="urn:navmap:repair:error",
        )
        json_str = render_cli_envelope(envelope)

        parsed = json.loads(json_str)
        assert "problem" in parsed
        assert parsed["problem"]["type"] == "https://kgfoundry.dev/problems/navmap-repair-error"
        assert parsed["problem"]["status"] == 500
