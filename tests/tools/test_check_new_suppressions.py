"""Tests for the suppression guard CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import pytest
import tools.check_new_suppressions as suppression_guard

from kgfoundry_common.errors import ConfigurationError


class SuppressionViolationEntry(TypedDict):
    line: int
    preview: str


class SuppressionFileEntry(TypedDict):
    file: str
    violations: list[SuppressionViolationEntry]


class SuppressionContext(TypedDict):
    violation_count: int
    files: list[SuppressionFileEntry]


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
    assert context is not None
    context_dict = cast("SuppressionContext", context)
    assert context_dict["violation_count"] == 1
    file_entry = context_dict["files"][0]
    file_path = Path(file_entry["file"])
    assert file_path.name == "module.py"
    first_violation = file_entry["violations"][0]
    assert first_violation["line"] == 1


def test_run_suppression_guard_allows_ticket_metadata(tmp_path: Path) -> None:
    """Files with ticket metadata should pass without raising."""
    _write(tmp_path, "module.py", "# type: ignore  # TICKET: TEST-1\n")

    suppression_guard.run_suppression_guard([tmp_path])


def test_resolve_target_directories_validates_input(tmp_path: Path) -> None:
    """Invalid directories should surface as configuration errors."""
    with pytest.raises(ConfigurationError):
        suppression_guard._resolve_target_directories(["./does-not-exist"])  # noqa: SLF001

    path = tmp_path / "not_a_dir.txt"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        suppression_guard._resolve_target_directories([str(path)])  # noqa: SLF001


def test_resolve_target_directories_accepts_existing_directory(tmp_path: Path) -> None:
    """Valid directories should be resolved and returned."""
    resolved = suppression_guard._resolve_target_directories([str(tmp_path)])  # noqa: SLF001

    assert resolved == [tmp_path.resolve()]
