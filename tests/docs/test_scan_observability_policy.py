"""Tests for observability policy loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tools.docs import scan_observability


@pytest.fixture(name="policy_file")
def fixture_policy_file(tmp_path: Path) -> Path:
    """Create a temporary policy file used by tests."""
    policy_path = tmp_path / "observability.yml"
    policy_path.write_text(
        textwrap.dedent(
            """
            metric:
              require_unit_suffix: false
            """
        ).strip(),
    )
    return policy_path


def test_load_policy_deep_merge_preserves_defaults(
    policy_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure overriding a nested key preserves defaults from the base policy."""
    monkeypatch.setattr(scan_observability, "POLICY_PATH", policy_file)

    policy = scan_observability.load_policy()

    assert policy["metric"]["require_unit_suffix"] is False
    assert (
        policy["metric"]["allowed_units"]
        == scan_observability.DEFAULT_POLICY["metric"]["allowed_units"]
    )
    assert policy["labels"] == scan_observability.DEFAULT_POLICY["labels"]
