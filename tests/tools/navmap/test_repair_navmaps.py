from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.navmap.build_navmap import ModuleInfo
from tools.navmap.migrate_navmaps import migrate_navmaps
from tools.navmap.repair_navmaps import RepairResult, repair_module


def _module_source() -> str:
    return '"""Example module."""\n__all__ = [\'foo\']\n\ndef foo() -> str:\n    return \'bar\'\n'


@pytest.fixture
def module_file(tmp_path: Path) -> Path:
    path = tmp_path / "example.py"
    path.write_text(_module_source(), encoding="utf-8")
    return path


def _module_info(path: Path) -> ModuleInfo:
    return ModuleInfo(
        module="pkg.example",
        path=path,
        exports=["foo"],
        sections={},
        anchors={},
        nav_sections=[],
        navmap_dict={},
    )


def test_repair_module_reports_changes_without_apply(module_file: Path) -> None:
    info = _module_info(module_file)

    result = repair_module(info, apply=False)

    assert isinstance(result, RepairResult)
    assert result.module == module_file
    assert result.changed is True
    assert result.applied is False
    assert any("navmap" in message for message in result.messages)
    assert module_file.read_text(encoding="utf-8") == _module_source()


def test_repair_module_applies_changes_to_disk(module_file: Path) -> None:
    info = _module_info(module_file)

    result = repair_module(info, apply=True)

    assert result.applied is True
    written = module_file.read_text(encoding="utf-8")
    assert "__navmap__" in written
    assert "[nav:anchor" in written


def test_migrate_navmaps_writes_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {"commit": "HEAD", "modules": {}}

    def _fake_build_index() -> dict[str, object]:
        return payload

    monkeypatch.setattr("tools.navmap.migrate_navmaps.build_index", _fake_build_index)

    output = tmp_path / "navmap.json"
    result = migrate_navmaps(output=output, pretty=True)

    assert result == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
