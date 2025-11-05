"""Regression tests for MkDocs interface catalog generation."""

from __future__ import annotations

import importlib
import io
import json
import sys
import types
from pathlib import Path

import pytest


class _DummyFile(io.StringIO):
    """Collect writes made through ``mkdocs_gen_files`` stubs."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def __enter__(self) -> "_DummyFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Store written content keyed by the path before closing."""
        _CAPTURED_OUTPUTS[self._path] = self.getvalue()
        return False


_CAPTURED_OUTPUTS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _clear_captured_outputs() -> None:
    _CAPTURED_OUTPUTS.clear()


def _install_mkdocs_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight ``mkdocs_gen_files`` shim for testing."""

    def _fake_open(path: object, *_, **__) -> _DummyFile:
        return _DummyFile(str(path))

    stub = types.SimpleNamespace(open=_fake_open, files=io.StringIO)
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", stub)


def test_render_interface_catalog_links_full_module_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interfaces nested under packages should link using the full dotted path."""

    _install_mkdocs_stub(monkeypatch)
    module_name = "tools.mkdocs_suite.docs._scripts.gen_interface_pages"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    _CAPTURED_OUTPUTS.clear()

    src_pkg = tmp_path / "src" / "pkg_root" / "subpkg" / "leaf"
    src_pkg.mkdir(parents=True)
    nav_payload = {
        "interfaces": [
            {
                "id": "pkg.interfaces.Example",
                "type": "service",
            }
        ]
    }
    (src_pkg / "_nav.json").write_text(json.dumps(nav_payload), encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "api_registry.yaml")

    module.render_interface_catalog()

    output = _CAPTURED_OUTPUTS.get("api/interfaces.md")
    assert output is not None
    assert "[pkg_root.subpkg.leaf](../modules/pkg_root.subpkg.leaf.md)" in output

    sys.modules.pop(module_name, None)
