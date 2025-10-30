"""Tests for the ``tools.docs.build_test_map`` CLI behavior."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from tools.docs import build_test_map


@pytest.fixture
def _reload_build_test_map() -> Iterator[None]:
    """Reload the build_test_map module to reset global state."""
    importlib.reload(build_test_map)
    try:
        yield
    finally:
        importlib.reload(build_test_map)


def _patch_output_paths(tmp_path: Path) -> None:
    """Point build_test_map outputs to a temporary directory."""
    outdir = tmp_path / "docs_build"
    outdir.mkdir(parents=True, exist_ok=True)
    build_test_map.OUTDIR = outdir
    build_test_map.OUTFILE_MAP = outdir / "test_map.json"
    build_test_map.OUTFILE_COV = outdir / "test_map_coverage.json"
    build_test_map.OUTFILE_SUM = outdir / "test_map_summary.json"
    build_test_map.OUTFILE_LINT = outdir / "test_map_lint.json"
    build_test_map.COV_JSON = outdir / "coverage.json"


def test_build_test_map_exits_when_navmap_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _reload_build_test_map: Iterator[None]
) -> None:
    """The CLI should exit non-zero when the navmap cannot be loaded."""
    _patch_output_paths(tmp_path)
    navmap = build_test_map.ROOT / "site" / "_build" / "navmap" / "navmap.json"
    backup = None
    if navmap.exists():
        backup = tmp_path / "navmap_backup.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        navmap.replace(backup)
    try:
        with pytest.raises(SystemExit) as exc_info:
            build_test_map.main()
    finally:
        if backup and backup.exists():
            navmap.parent.mkdir(parents=True, exist_ok=True)
            backup.replace(navmap)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "navmap.json" in captured.err
    assert "ERROR" in captured.err
