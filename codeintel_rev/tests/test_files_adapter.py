"""Unit tests for file adapter path handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel_rev.mcp_server.adapters import files as files_adapter
from codeintel_rev.mcp_server.service_context import reset_service_context


def _expect(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _set_repo_root(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setenv("REPO_ROOT", str(repo_root))
    # Ensure cached service context does not hold stale settings across tests.
    reset_service_context()


def test_open_file_rejects_paths_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_file should reject attempts to escape the repository root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "README.md").write_text("within repo", encoding="utf-8")

    _set_repo_root(monkeypatch, repo_root)

    result = files_adapter.open_file("../outside.txt")
    _expect(
        condition=result.get("error", "").startswith("Path"),
        message="Expected path traversal to be rejected",
    )
    reset_service_context()


def test_open_file_rejects_negative_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_file should error when provided negative line bounds."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "file.txt").write_text("line1\nline2\n", encoding="utf-8")

    _set_repo_root(monkeypatch, repo_root)

    result = files_adapter.open_file("file.txt", start_line=-1, end_line=2)
    _expect(
        condition="error" in result,
        message="Expected an error payload for negative bounds",
    )
    _expect(
        condition="positive integer" in result["error"],
        message="Expected positive integer error message",
    )
    _expect(
        condition="content" not in result,
        message="Expected response to omit content when failing",
    )
    reset_service_context()


def test_open_file_rejects_start_line_after_end_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_file should error when start_line exceeds end_line."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "file.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")

    _set_repo_root(monkeypatch, repo_root)

    result = files_adapter.open_file("file.txt", start_line=3, end_line=2)
    _expect(
        condition="error" in result,
        message="Expected an error payload when start_line > end_line",
    )
    _expect(
        condition="less than or equal" in result["error"],
        message="Expected start_line <= end_line message",
    )
    _expect(
        condition="content" not in result,
        message="Expected response to omit content when failing",
    )
    reset_service_context()


def test_list_paths_rejects_paths_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_paths should reject traversal outside the repository root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "inner").mkdir()
    (repo_root / "inner" / "file.txt").write_text("data", encoding="utf-8")

    _set_repo_root(monkeypatch, repo_root)

    response = files_adapter.list_paths(path="../inner")
    _expect(
        condition=bool(response.get("error")),
        message="Expected traversal attempt to return an error",
    )
    _expect(
        condition="escapes repository root" in response["error"],
        message="Expected repository escape message",
    )
    reset_service_context()
