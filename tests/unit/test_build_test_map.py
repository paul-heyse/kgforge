from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "docs" / "build_test_map.py"
SPEC = importlib.util.spec_from_file_location("build_test_map_under_test", MODULE_PATH)
assert SPEC and SPEC.loader  # pragma: no cover - hard failure if module cannot be loaded
build_test_map = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_test_map)


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
"""Tests for ``tools.docs.build_test_map`` utilities."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "docs" / "build_test_map.py"
SPEC = importlib.util.spec_from_file_location("build_test_map", MODULE_PATH)
assert SPEC and SPEC.loader
_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_MODULE)  # type: ignore[arg-type]


def _normalize_repo_rel(path_like: str) -> str:
    return getattr(_MODULE, "_normalize_repo_rel")(path_like)


ROOT = getattr(_MODULE, "ROOT")


def test_normalize_repo_rel_with_prefixed_root() -> None:
    """Absolute paths containing the repo root collapse to repo-relative."""

    dummy_rel_path = Path("src") / "dummy.py"
    prefixed_path = os.path.join("/tmp/cache", str(ROOT).lstrip(os.sep), *dummy_rel_path.parts)

    normalized = _normalize_repo_rel(prefixed_path)

    assert normalized == str(dummy_rel_path)
