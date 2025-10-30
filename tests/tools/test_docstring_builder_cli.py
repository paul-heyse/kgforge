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


@pytest.mark.parametrize(
    ("subcommand", "handler_name"),
    [
        ("generate", "_command_generate"),
        ("fix", "_command_fix"),
        ("diff", "_command_diff"),
        ("check", "_command_check"),
        ("lint", "_command_lint"),
        ("measure", "_command_measure"),
        ("schema", "_command_schema"),
        ("doctor", "_command_doctor"),
        ("list", "_command_list"),
        ("clear-cache", "_command_clear_cache"),
        ("harvest", "_command_harvest"),
    ],
)
def test_main_dispatches_subcommands(
    subcommand: str, handler_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_handler(args: argparse.Namespace) -> int:
        calls.append(subcommand)
        return 0

    monkeypatch.setattr(cli, handler_name, fake_handler)
    exit_code = cli.main([subcommand])
    assert exit_code == 0
    assert calls == [subcommand]


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


def _make_args(
    **overrides: bool | list[str] | str | None,
) -> argparse.Namespace:
    base: dict[str, bool | list[str] | str | None] = {
        "paths": [],
        "module": "",
        "since": "",
        "force": False,
        "diff": False,
        "ignore_missing": False,
        "changed_only": False,
        "only_plugin": [],
        "disable_plugin": [],
        "policy_override": [],
        "subcommand": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_select_files_rejects_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    config = BuilderConfig()
    args = _make_args(paths=[str(outside)])
    with pytest.raises(cli.InvalidPathError):
        list(cli._select_files(config, args))


def test_select_files_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path.parent / "target.py"
    target.write_text("print('target')\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    link.symlink_to(target)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    config = BuilderConfig()
    args = _make_args(paths=[str(link)])
    with pytest.raises(cli.InvalidPathError):
        list(cli._select_files(config, args))


def test_doctor_invokes_stub_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "stubs" / "griffe").mkdir(parents=True)
    (tmp_path / "stubs" / "libcst").mkdir(parents=True)
    (tmp_path / "stubs" / "mkdocs_gen_files").mkdir(parents=True)
    (tmp_path / "docs" / "_build").mkdir(parents=True)
    (tmp_path / ".cache").mkdir(parents=True)
    (tmp_path / "mypy.ini").write_text("[mypy]\nmypy_path = src:stubs\n", encoding="utf-8")
    precommit = {
        "repos": [
            {
                "repo": "local",
                "hooks": [
                    {"name": "docstring-builder (check)"},
                    {"name": "docs: regenerate artifacts"},
                    {"name": "navmap-check"},
                    {"name": "pyrefly-check"},
                ],
            }
        ]
    }
    import yaml

    (tmp_path / ".pre-commit-config.yaml").write_text(yaml.safe_dump(precommit), encoding="utf-8")

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "run_stub_drift", lambda: 1)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: object())

    exit_code = cli.main(["doctor", "--stubs"])
    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_CONFIG
    assert "stub drift" in output.lower()
