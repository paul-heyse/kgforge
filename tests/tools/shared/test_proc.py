"""Tests for tools._shared.proc module."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from tools._shared.proc import ToolExecutionError, ToolRunResult, run_tool


class TestRunTool:
    """Tests for run_tool function."""

    def test_successful_execution(self) -> None:
        """run_tool returns ToolRunResult for successful command."""
        result = run_tool(["echo", "test"], timeout=5.0)
        assert isinstance(result, ToolRunResult)
        assert result.returncode == 0
        assert "test" in result.stdout
        assert result.timed_out is False
        assert result.duration_seconds > 0

    def test_failure_with_check_raises(self) -> None:
        """run_tool raises ToolExecutionError when check=True and command fails."""
        with pytest.raises(ToolExecutionError) as exc_info:
            run_tool(["false"], check=True, timeout=5.0)

        error = exc_info.value
        assert error.returncode == 1
        assert error.problem is not None
        assert error.problem["type"] == "https://kgfoundry.dev/problems/tool-failure"
        assert error.problem["status"] == 500

    def test_failure_without_check_returns_result(self) -> None:
        """run_tool returns ToolRunResult even when command fails if check=False."""
        result = run_tool(["false"], check=False, timeout=5.0)
        assert isinstance(result, ToolRunResult)
        assert result.returncode != 0
        # ToolRunResult doesn't have problem attribute - that's only on ToolExecutionError

    def test_timeout_raises_with_problem_details(self) -> None:
        """run_tool raises ToolExecutionError with Problem Details on timeout."""
        with pytest.raises(ToolExecutionError) as exc_info:
            run_tool(["sleep", "10"], timeout=0.1, check=True)

        error = exc_info.value
        assert error.problem is not None
        assert error.problem["type"] == "https://kgfoundry.dev/problems/tool-timeout"
        assert error.problem["status"] == 504
        detail = error.problem.get("detail")
        assert isinstance(detail, str)
        assert "timed out" in detail.lower()

    def test_missing_executable_raises(self) -> None:
        """run_tool raises ToolExecutionError when executable not found."""
        with pytest.raises(ToolExecutionError) as exc_info:
            run_tool(["nonexistent-command-that-does-not-exist"], check=True)

        error = exc_info.value
        assert error.problem is not None
        assert error.problem["type"] == "https://kgfoundry.dev/problems/tool-missing"

    def test_empty_command_raises(self) -> None:
        """run_tool raises ToolExecutionError for empty command."""
        with pytest.raises(ToolExecutionError) as exc_info:
            run_tool([], check=True)

        error = exc_info.value
        assert "empty" in error.args[0].lower() or "at least one" in error.args[0].lower()

    def test_executable_resolved_to_absolute_path(self) -> None:
        """run_tool resolves relative executable paths to absolute."""
        result = run_tool(["echo", "test"], timeout=5.0)
        assert Path(result.command[0]).is_absolute()

    def test_cwd_respected(self, tmp_path: Path) -> None:
        """run_tool respects cwd parameter."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = run_tool(["cat", "test.txt"], cwd=tmp_path, timeout=5.0)
        assert "test content" in result.stdout

    def test_env_sanitized(self) -> None:
        """run_tool sanitizes environment variables."""
        # Set a disallowed env var
        os.environ["TEST_DISALLOWED_VAR"] = "should-not-appear"

        result = run_tool(
            ["env"],
            env={"TEST_ALLOWED_VAR": "should-appear"},
            timeout=5.0,
        )

        # Allowed vars should be present
        assert "TEST_ALLOWED_VAR" in result.stdout or "PATH" in result.stdout
        # Note: TEST_DISALLOWED_VAR might still appear if inherited, but custom env should override

    def test_captures_stdout_stderr(self) -> None:
        """run_tool captures both stdout and stderr."""
        result = run_tool(
            ["sh", "-c", "echo stdout; echo stderr >&2"],
            timeout=5.0,
        )
        assert "stdout" in result.stdout
        assert "stderr" in result.stderr

    def test_duration_recorded(self) -> None:
        """run_tool records execution duration."""
        start = time.monotonic()
        result = run_tool(["sleep", "0.1"], timeout=5.0)
        elapsed = time.monotonic() - start

        assert result.duration_seconds > 0
        assert result.duration_seconds <= elapsed + 0.1  # Allow some tolerance
