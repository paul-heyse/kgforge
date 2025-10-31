"""Tests for the docstring builder CLI helpers."""

from __future__ import annotations

import importlib

import pytest
from tools.docstring_builder import cli
from tools.docstring_builder.models import (
    ErrorReport,
    RunStatus,
    build_cli_result_skeleton,
    validate_cli_output,
)


@pytest.mark.parametrize(
    ("exit_status", "expect_problem"),
    [
        (cli.ExitStatus.SUCCESS, False),
        (cli.ExitStatus.VIOLATION, True),
        (cli.ExitStatus.ERROR, True),
    ],
)
def test_cli_problem_details(exit_status: cli.ExitStatus, expect_problem: bool) -> None:
    """Problem Details envelopes are attached for non-successful runs."""
    cli_result = build_cli_result_skeleton(cli._status_from_exit(exit_status))
    cli_result["command"] = "update"
    cli_result["subcommand"] = "check"
    cli_result["summary"]["subcommand"] = "check"
    cli_result["files"] = []
    errors: list[ErrorReport] = [
        {
            "file": "tools/example.py",
            "status": RunStatus.ERROR,
            "message": "Unable to render docstring",
        }
    ]
    cli_result["errors"] = errors
    problem = cli._build_problem_details(
        exit_status,
        errors,
        command="update",
        subcommand="check",
    )
    if problem is not None:
        cli_result["problem"] = problem
    validate_cli_output(cli_result)
    assert ("problem" in cli_result) is expect_problem
    if expect_problem:
        assert cli_result["problem"]["status"] == cli._http_status_for_exit(exit_status)


@pytest.mark.parametrize(
    ("flag_value", "expected"),
    [
        ("1", True),
        ("0", False),
        ("false", False),
        ("yes", True),
    ],
)
def test_typed_pipeline_toggle(
    monkeypatch: pytest.MonkeyPatch, flag_value: str, expected: bool
) -> None:
    """Environment flag toggles the typed pipeline helper."""
    monkeypatch.setenv("DOCSTRINGS_TYPED_IR", flag_value)
    reloaded = importlib.reload(cli)
    assert reloaded._typed_pipeline_enabled() is expected
