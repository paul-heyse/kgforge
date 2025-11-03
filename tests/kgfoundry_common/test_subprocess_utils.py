"""Tests for subprocess utilities with timeout and safety features."""

from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING

import pytest

from kgfoundry_common.subprocess_utils import (
    SubprocessError,
    SubprocessTimeoutError,
    run_subprocess,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestSubprocessExecution:
    """Test suite for subprocess execution."""

    def test_simple_command_success(self) -> None:
        """Test successful simple command execution."""
        result = run_subprocess(["echo", "hello"], timeout=5)
        assert "hello" in result

    def test_command_with_output(self) -> None:
        """Test command that produces output."""
        result = run_subprocess(
            ["python", "-c", "print('test output')"],
            timeout=10,
        )
        assert "test output" in result

    def test_command_failure_raises_error(self) -> None:
        """Test that failed command raises SubprocessError."""
        with pytest.raises(SubprocessError) as exc_info:
            run_subprocess(
                ["python", "-c", "import sys; sys.exit(1)"],
                timeout=5,
            )
        assert exc_info.value.returncode == 1

    def test_command_with_stderr(self) -> None:
        """Test that stderr is captured in error."""
        with pytest.raises(SubprocessError) as exc_info:
            run_subprocess(
                ["python", "-c", "import sys; sys.stderr.write('error'); sys.exit(1)"],
                timeout=5,
            )
        assert exc_info.value.stderr is not None

    def test_timeout_enforcement(self) -> None:
        """Test that timeout is enforced."""
        with pytest.raises(SubprocessTimeoutError):
            run_subprocess(
                ["python", "-c", "import time; time.sleep(10)"],
                timeout=1,
            )

    def test_timeout_error_has_details(self) -> None:
        """Test that timeout error contains useful details."""
        with pytest.raises(SubprocessTimeoutError) as exc_info:
            run_subprocess(
                ["sleep", "10"],
                timeout=1,
            )
        assert exc_info.value.timeout_seconds == 1
        assert exc_info.value.command is not None

    def test_invalid_timeout_too_low(self) -> None:
        """Test that timeout < 1 is rejected."""
        with pytest.raises(ValueError, match="between"):
            run_subprocess(["echo", "test"], timeout=0)

    def test_invalid_timeout_too_high(self) -> None:
        """Test that timeout > 3600 is rejected."""
        with pytest.raises(ValueError, match="between"):
            run_subprocess(["echo", "test"], timeout=3601)

    def test_valid_timeout_boundaries(self) -> None:
        """Test that boundary timeout values are accepted."""
        # Minimum valid timeout
        result = run_subprocess(["echo", "test"], timeout=1)
        assert "test" in result

        # Maximum valid timeout
        result = run_subprocess(["echo", "test"], timeout=3600)
        assert "test" in result

    def test_timeout_none_uses_default(self) -> None:
        """Test that None timeout uses default."""
        result = run_subprocess(["echo", "test"], timeout=None)
        assert "test" in result


class TestWorkingDirectory:
    """Test suite for working directory handling."""

    def test_cwd_is_used(self, tmp_path: Path) -> None:
        """Test that cwd parameter is used."""
        # Create a test file in tmp_path
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # List files in that directory
        result = run_subprocess(
            ["ls", "test.txt"],
            timeout=5,
            cwd=tmp_path,
        )
        assert "test.txt" in result

    def test_cwd_resolved_to_absolute_path(self, tmp_path: Path) -> None:
        """Test that cwd is resolved to absolute path."""
        # Pass a relative-looking path (after resolution it's absolute)
        result = run_subprocess(
            ["pwd"],
            timeout=5,
            cwd=tmp_path,
        )
        # Should contain the absolute path
        assert str(tmp_path.resolve()) in result or str(tmp_path) in result

    def test_cwd_none_works(self) -> None:
        """Test that cwd=None (inherit parent cwd) works."""
        result = run_subprocess(["pwd"], timeout=5, cwd=None)
        assert "/" in result  # Should output a path


class TestEnvironmentVariables:
    """Test suite for environment variable handling."""

    def test_env_dict_passed(self) -> None:
        """Test that environment dict is passed to subprocess."""
        env = {"TEST_VAR": "test_value", "PATH": "/usr/bin:/bin"}
        result = run_subprocess(
            ["python", "-c", "import os; print(os.environ.get('TEST_VAR'))"],
            timeout=5,
            env=env,
        )
        assert "test_value" in result

    def test_env_none_inherits_parent(self) -> None:
        """Test that env=None inherits parent environment."""
        result = run_subprocess(
            ["python", "-c", "import os; print(bool(os.environ.get('PATH')))"],
            timeout=5,
            env=None,
        )
        assert "True" in result

    def test_env_dict_mapping(self) -> None:
        """Test that env can be any Mapping."""
        env = UserDict({"CUSTOM": "value", "PATH": "/bin"})
        result = run_subprocess(
            ["python", "-c", "import os; print(os.environ.get('CUSTOM'))"],
            timeout=5,
            env=env,
        )
        assert "value" in result


class TestErrorReporting:
    """Test suite for error messages."""

    def test_subprocess_error_message_includes_command(self) -> None:
        """Test that error message includes command."""
        with pytest.raises(SubprocessError) as exc_info:
            run_subprocess(["false"], timeout=5)
        assert "false" in str(exc_info.value)

    def test_timeout_error_message_includes_command(self) -> None:
        """Test that timeout error includes command."""
        with pytest.raises(SubprocessTimeoutError) as exc_info:
            run_subprocess(["sleep", "10"], timeout=1)
        assert "sleep" in str(exc_info.value)


class TestComplexCommands:
    """Test suite for complex command scenarios."""

    def test_python_script_with_args(self) -> None:
        """Test Python script with arguments."""
        result = run_subprocess(
            [
                "python",
                "-c",
                "import sys; print(f'{len(sys.argv)} {sys.argv[1]}')",
                "arg1",
            ],
            timeout=5,
        )
        assert "arg1" in result

    def test_shell_command_via_python(self) -> None:
        """Test shell-like commands via Python."""
        result = run_subprocess(
            [
                "python",
                "-c",
                ("data = [1, 2, 3]; print(','.join(map(str, data)))"),
            ],
            timeout=5,
        )
        assert "1,2,3" in result

    @pytest.mark.parametrize(
        "value",
        ["test", "hello world", "123", ""],
    )
    def test_echo_various_inputs(self, value: str) -> None:
        """Test echo command with various inputs."""
        result = run_subprocess(["echo", value], timeout=5)
        if value:
            assert value in result
