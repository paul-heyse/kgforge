from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest
from tools.docstring_builder import cli
from tools.docstring_builder.cache import BuilderCache
from tools.docstring_builder.config import BuilderConfig


def test_main_translates_legacy_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_check(args: argparse.Namespace) -> int:
        calls.append("check")
        return 0

    monkeypatch.setattr(cli, "_command_check", fake_check)
    exit_code = cli.main(["--diff", "--since", "HEAD"])
    assert exit_code == 0
    assert calls == ["check"]


def test_process_file_ignore_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "docs" / "_build" / "example.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("print('example')\n", encoding="utf-8")

    config = BuilderConfig()
    cache = BuilderCache(tmp_path / "cache.json")

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    module_name = "docs._build.example"

    def fake_harvest(path: Path, _config: BuilderConfig, _root: Path) -> object:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(cli, "harvest_file", fake_harvest)

    options = cli.ProcessingOptions(
        command="update",
        force=False,
        ignore_missing=True,
        missing_patterns=tuple(cli.MISSING_MODULE_PATTERNS),
        skip_docfacts=False,
    )
    result = cli._process_file(
        file_path,
        config,
        cache,
        options,
        plugin_manager=None,
    )
    assert result.status == cli.ExitStatus.SUCCESS
    assert result.docfacts == []
    assert result.preview is None
    assert result.changed is False
    assert result.skipped is True

    options_error = dataclasses.replace(options, ignore_missing=False)
    result_error = cli._process_file(
        file_path,
        config,
        cache,
        options_error,
        plugin_manager=None,
    )
    assert result_error.status == cli.ExitStatus.CONFIG
