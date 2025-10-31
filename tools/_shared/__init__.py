"""Shared utilities for tooling packages."""

from __future__ import annotations

from tools._shared.logging import StructuredLoggerAdapter, get_logger, with_fields
from tools._shared.problem_details import (
    ProblemDetailsDict,
    build_problem_details,
    render_problem,
)
from tools._shared.proc import ToolExecutionError, ToolRunResult, run_tool

__all__ = [
    "ProblemDetailsDict",
    "StructuredLoggerAdapter",
    "ToolExecutionError",
    "ToolRunResult",
    "build_problem_details",
    "get_logger",
    "render_problem",
    "run_tool",
    "with_fields",
]
