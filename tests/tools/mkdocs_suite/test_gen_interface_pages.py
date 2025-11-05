"""Tests for the interface catalog generation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tools.mkdocs_suite.docs._scripts import gen_interface_pages


@pytest.fixture(name="temporary_repo")
def fixture_temporary_repo(tmp_path: Path) -> Path:
    """Create a temporary repository layout for nav discovery tests."""

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "valid").mkdir(parents=True)
    (repo_root / "src" / "invalid").mkdir(parents=True)
    valid_nav = repo_root / "src" / "valid" / "_nav.json"
    invalid_nav = repo_root / "src" / "invalid" / "_nav.json"

    valid_payload = {"interfaces": [{"id": "valid-interface"}]}
    valid_nav.write_text(json.dumps(valid_payload), encoding="utf-8")
    invalid_nav.write_text("{not-json", encoding="utf-8")

    return repo_root


def test_collect_nav_interfaces_skips_malformed_json(
    temporary_repo: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed nav files should be ignored without raising errors."""

    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(gen_interface_pages, "REPO_ROOT", temporary_repo)

    interfaces = gen_interface_pages._collect_nav_interfaces()

    assert any("invalid/_nav.json" in record.message for record in caplog.records)
    assert interfaces == [{"id": "valid-interface", "module": "valid"}]
