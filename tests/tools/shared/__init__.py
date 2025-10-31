"""Tests for tools._shared module integration."""

from __future__ import annotations

import pytest
from tools._shared import (
    ProblemDetailsDict,
    StructuredLoggerAdapter,
    ToolExecutionError,
    ToolRunResult,
    build_problem_details,
    get_logger,
    render_problem,
    run_tool,
    with_fields,
)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_symbols_exported(self) -> None:
        """All expected symbols are exported from tools._shared."""
        # Verify imports work
        assert ProblemDetailsDict is not None
        assert StructuredLoggerAdapter is not None
        assert ToolExecutionError is not None
        assert ToolRunResult is not None
        assert build_problem_details is not None
        assert get_logger is not None
        assert render_problem is not None
        assert run_tool is not None
        assert with_fields is not None

    def test_end_to_end_workflow(self) -> None:
        """Complete workflow: logging, subprocess, problem details."""
        # Get logger
        logger = get_logger(__name__)
        assert logger is not None

        # Run a successful command
        result = run_tool(["echo", "test"], timeout=5.0)
        assert isinstance(result, ToolRunResult)
        assert result.returncode == 0

        # Verify error handling produces Problem Details
        with pytest.raises(ToolExecutionError) as exc_info:
            run_tool(["false"], check=True, timeout=5.0)

        error = exc_info.value
        assert error.problem is not None

        # Render Problem Details
        json_str = render_problem(error.problem)
        assert isinstance(json_str, str)
        assert "tool-failure" in json_str
