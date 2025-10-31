from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from tools import gen_readmes
from tools.gen_readmes import (
    EditorMode,
    LinkMode,
    NavData,
    NavModuleData,
    ReadmeConfig,
    SymbolMetadata,
    TestCatalog,
    TestRecord,
    badges_for,
    write_if_changed,
)


def test_badges_for_merges_default_and_override() -> None:
    module = NavModuleData(
        identifier="pkg.mod",
        defaults=SymbolMetadata(owner="docs", stability="stable"),
        overrides={
            "pkg.mod.symbol": SymbolMetadata(owner="override", section="api"),
        },
        sections={"symbol": "api"},
        listed_symbols=frozenset({"symbol"}),
    )
    nav = NavData(modules={module.identifier: module})
    tests = TestCatalog(
        records={
            "pkg.mod.symbol": (TestRecord(file="tests/test_readmes.py", lines=(42,)),),
        }
    )

    badge = badges_for("pkg.mod.symbol", nav=nav, tests=tests)

    assert badge.owner == "override"
    assert badge.stability == "stable"
    assert badge.section == "api"
    assert badge.tested_by[0].file == "tests/test_readmes.py"
    assert badge.tested_by[0].lines == (42,)


def test_format_test_badge_handles_records() -> None:
    entries = (
        TestRecord(file="tests/test_alpha.py", lines=(10, 11)),
        TestRecord(file="tests/test_beta.py"),
    )

    snippet = gen_readmes._format_test_badge(entries)

    assert snippet == "`tested-by: tests/test_alpha.py:10, tests/test_beta.py`"


def test_write_if_changed_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    content = "# heading\n"

    first_write = write_if_changed(path, content)
    second_write = write_if_changed(path, content)

    assert first_write is True
    assert second_write is False
    rendered = path.read_text(encoding="utf-8")
    assert "agent:readme" in rendered


def test_readme_config_from_namespace_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.gen_readmes.iter_packages",
        lambda: ["pkg_one", "pkg_two"],
    )
    namespace = argparse.Namespace(
        packages="",
        link_mode="both",
        editor="relative",
        fail_on_metadata_miss=True,
        dry_run=False,
        verbose=True,
        run_doctoc=False,
    )

    config = ReadmeConfig.from_namespace(namespace)

    assert config.packages == ("pkg_one", "pkg_two")
    assert config.link_mode is LinkMode.BOTH
    assert config.editor is EditorMode.RELATIVE
    assert config.fail_on_metadata_miss is True
