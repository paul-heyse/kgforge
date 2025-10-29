"""Tests for pruning stale schema exports."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from pytest import MonkeyPatch


def _load_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tests.docs.export_schemas_pruning", Path("tools/docs/export_schemas.py")
    )
    if spec is None or spec.loader is None:
        message = "Unable to load export_schemas module for testing"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _prepare_environment(monkeypatch: MonkeyPatch, tmp_path: Path) -> tuple[ModuleType, Path, Path]:
    """Configure the exporter module to operate within ``tmp_path``."""
    export_schemas = _load_exporter()
    out_dir = tmp_path / "schemas"
    out_dir.mkdir()
    drift_out = tmp_path / "schema_drift.json"

    monkeypatch.setattr(export_schemas, "OUT", out_dir, raising=False)
    monkeypatch.setattr(export_schemas, "DRIFT_OUT", drift_out, raising=False)
    monkeypatch.setattr(export_schemas, "NAVMAP", tmp_path / "navmap.json", raising=False)
    monkeypatch.setattr(export_schemas, "_load_navmap", lambda: {}, raising=False)
    monkeypatch.setattr(export_schemas, "_iter_models", lambda: iter(()), raising=False)

    return export_schemas, out_dir, drift_out


def test_prunes_stale_schema_files(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Stale schemas are removed and recorded in drift summaries."""
    export_schemas, out_dir, drift_out = _prepare_environment(monkeypatch, tmp_path)
    stale_path = out_dir / "stale.Model.json"
    stale_path.write_text(json.dumps({"title": "Old"}) + "\n", encoding="utf-8")

    exit_code = export_schemas.main([])

    assert exit_code == 0
    assert not stale_path.exists()

    drift = json.loads(drift_out.read_text(encoding="utf-8"))
    assert str(stale_path) in drift
    assert drift[str(stale_path)]["top_level_removed"] == ["title"]


def test_check_drift_flags_stale_files(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """The --check-drift flag surfaces stale files without deleting them."""
    export_schemas, out_dir, drift_out = _prepare_environment(monkeypatch, tmp_path)
    stale_path = out_dir / "stale.Model.json"
    stale_path.write_text(json.dumps({"title": "Old"}) + "\n", encoding="utf-8")

    exit_code = export_schemas.main(["--check-drift"])

    assert exit_code == 2
    assert stale_path.exists()

    drift = json.loads(drift_out.read_text(encoding="utf-8"))
    assert str(stale_path) in drift
    assert drift[str(stale_path)]["top_level_removed"] == ["title"]
