"""Tests for the suppression guard CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tools.check_new_suppressions as suppression_guard

from kgfoundry_common.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


def _write(base: Path, name: str, content: str) -> Path:
    path = base / name
    path.write_text(content, encoding="utf-8")
    return path


def test_run_suppression_guard_detects_missing_ticket(tmp_path: Path) -> None:
    """The guard should raise when a suppression lacks ``TICKET:`` metadata."""
    _write(tmp_path, "module.py", "# type: ignore\n")

    with pytest.raises(ConfigurationError) as excinfo:
        suppression_guard.run_suppression_guard([tmp_path])

    context = excinfo.value.context
    expected_report = suppression_guard.check_directory(tmp_path)
    assert context == suppression_guard.build_guard_context(expected_report)


def test_run_suppression_guard_allows_ticket_metadata(tmp_path: Path) -> None:
    """Files with ticket metadata should pass without raising."""
    _write(tmp_path, "module.py", "# type: ignore  # TICKET: TEST-1\n")

    report = suppression_guard.run_suppression_guard([tmp_path])
    assert report.is_clean
    assert report.violation_count == 0


def test_resolve_target_directories_validates_input(tmp_path: Path) -> None:
    """Invalid directories should surface as configuration errors."""
    with pytest.raises(ConfigurationError):
        suppression_guard.resolve_target_directories(["./does-not-exist"])

    path = tmp_path / "not_a_dir.txt"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        suppression_guard.resolve_target_directories([str(path)])


def test_resolve_target_directories_accepts_existing_directory(tmp_path: Path) -> None:
    """Valid directories should be resolved and returned."""
    resolved = suppression_guard.resolve_target_directories([str(tmp_path)])

    assert resolved == [tmp_path.resolve()]
