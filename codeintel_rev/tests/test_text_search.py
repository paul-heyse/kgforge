"""Unit tests for text search adapter."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from codeintel_rev.mcp_server.adapters import text_search


def _build_match_line(path: Path) -> str:
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": str(path)},
                "line_number": 1,
                "lines": {"text": "example"},
                "submatches": [{"start": 0, "end": 1}],
            },
        }
    )


def test_search_text_flag_prefixed_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queries starting with a dash should be passed after `--`."""

    captured_args: list[list[str]] = []
    repo_root = Path.cwd()
    target_path = repo_root / "pyproject.toml"

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured_args.append(list(args))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=_build_match_line(target_path),
            stderr="",
        )

    monkeypatch.setattr(text_search.subprocess, "run", fake_run)

    result = text_search.search_text(
        "-def",
        regex=False,
        case_sensitive=False,
        paths=["pyproject.toml"],
        max_results=1,
    )

    assert captured_args, "The ripgrep process was not invoked"
    args = captured_args[0]
    sentinel_index = args.index("--")
    assert args[sentinel_index + 1] == "-def"
    assert result["matches"]
    assert result["matches"][0]["path"].endswith("pyproject.toml")


def test_search_text_returns_error_on_ripgrep_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ripgrep failures (return code > 1) should surface an error."""

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 2, stdout="", stderr="rg failed")

    monkeypatch.setattr(text_search.subprocess, "run", fake_run)

    result = text_search.search_text("pattern")

    assert result["matches"] == []
    assert result["total"] == 0
    assert result["error"] == "rg failed"


def test_fallback_grep_flag_prefixed_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grep fallback should also place the query after `--`."""

    captured_args: list[list[str]] = []
    repo_root = Path.cwd()
    target_path = repo_root / "pyproject.toml"

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured_args.append(list(args))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=f"{target_path}:1:example",
            stderr="",
        )

    monkeypatch.setattr(text_search.subprocess, "run", fake_run)

    result = text_search._fallback_grep(repo_root, "-def", case_sensitive=False, max_results=1)

    assert captured_args, "The grep process was not invoked"
    args = captured_args[0]
    sentinel_index = args.index("--")
    assert args[sentinel_index + 1] == "-def"
    assert result["matches"]
    assert result["matches"][0]["path"].endswith("pyproject.toml")
