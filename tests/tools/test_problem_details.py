from __future__ import annotations

from pathlib import Path

from tools import (
    build_tool_problem_details,
    tool_disallowed_problem_details,
    tool_failure_problem_details,
    tool_missing_problem_details,
    tool_timeout_problem_details,
)


def test_tool_timeout_problem_details_includes_command_and_timeout() -> None:
    problem = tool_timeout_problem_details(["git", "status"], timeout=10.0)

    assert problem["type"] == "https://kgfoundry.dev/problems/tool-timeout"
    assert problem["instance"] == "urn:tool:git:timeout"
    assert problem["command"] == ["git", "status"]
    assert problem["timeout"] == 10.0


def test_tool_failure_problem_details_marks_exit_code() -> None:
    problem = tool_failure_problem_details(["ruff"], returncode=3, detail="Lint error")

    assert problem["returncode"] == 3
    assert problem["instance"] == "urn:tool:ruff:exit-3"


def test_tool_disallowed_problem_details_sets_allowlist() -> None:
    allowlist = ["python*", "uv"]
    problem = tool_disallowed_problem_details(
        ["python", "-m", "module"], executable=Path("/usr/bin/python"), allowlist=allowlist
    )

    assert problem["type"] == "https://kgfoundry.dev/problems/tool-exec-disallowed"
    assert problem["allowlist"] == allowlist


def test_tool_missing_problem_details_uses_fallback_command() -> None:
    problem = tool_missing_problem_details([], executable="git", detail="git not found")

    assert problem["instance"] == "urn:tool:git:missing"
    assert problem["command"] == ["git"]


def test_build_tool_problem_details_accepts_extra_extensions() -> None:
    problem = build_tool_problem_details(
        category="tool-custom",
        command=["custom"],
        status=500,
        title="Custom failure",
        detail="Something broke",
        instance_suffix="custom",
        extensions={"foo": "bar"},
    )

    assert problem["foo"] == "bar"
    assert problem["command"] == ["custom"]
