"""Tests for ``tools.docs.build_test_map`` utilities."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "docs" / "build_test_map.py"

spec = importlib.util.spec_from_file_location("tools.docs.build_test_map", MODULE_PATH)
assert spec and spec.loader  # pragma: no cover - module must load for tests
build_test_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_test_map)


def test_load_symbol_candidates_strips_init(tmp_path, monkeypatch):
    root = tmp_path
    src = root / "src"
    pkg = src / "kgfoundry"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(build_test_map, "ROOT", root)
    monkeypatch.setattr(build_test_map, "SRC", src)

    candidates = build_test_map.load_symbol_candidates()

    assert "kgfoundry" in candidates
    assert "kgfoundry.__init__" not in candidates


def test_load_symbol_candidates_reads_navmap(tmp_path, monkeypatch):
    root = tmp_path
    navmap_dir = root / "site" / "_build" / "navmap"
    navmap_dir.mkdir(parents=True, exist_ok=True)
    navmap = navmap_dir / "navmap.json"
    navmap.write_text(
        json.dumps(
            {
                "modules": {
                    "brand_new": {"exports": ["Alpha", "brand_new.beta"]},
                    "legacy.__init__": {"exports": ["Gamma"]},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(build_test_map, "ROOT", root)
    monkeypatch.setattr(build_test_map, "SRC", root / "src")

    candidates = build_test_map.load_symbol_candidates()

    assert "brand_new" in candidates
    assert "brand_new.Alpha" in candidates
    assert "brand_new.beta" in candidates
    assert "legacy" in candidates
    assert "legacy.Gamma" in candidates


def _normalize_repo_rel(path_like: str) -> str:
    return build_test_map._normalize_repo_rel(path_like)


ROOT = build_test_map.ROOT


def test_normalize_repo_rel_with_prefixed_root() -> None:
    """Absolute paths containing the repo root collapse to repo-relative."""
    dummy_rel_path = Path("src") / "dummy.py"
    prefixed_path = os.path.join("/tmp/cache", str(ROOT).lstrip(os.sep), *dummy_rel_path.parts)

    normalized = _normalize_repo_rel(prefixed_path)

    assert normalized == str(dummy_rel_path)
