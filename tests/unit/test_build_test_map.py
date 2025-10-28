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
