from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_MODULE_CLASS = types.new_class("Module", (), {})


@pytest.fixture(name="module")
def fixture_module() -> types.ModuleType:
    """Return the module under test reloaded for isolation."""

    module_name = "tools.mkdocs_suite.docs._scripts.gen_module_pages"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _make_stub_module(name: str, module_path: Path) -> object:
    """Return a lightweight Griffe module stub for ``name``."""

    instance = _MODULE_CLASS()
    instance.path = name
    instance.members = {}
    instance.docstring = None
    instance.exports = None
    instance.relative_filepath = module_path
    return instance


def test_collect_modules_discovers_new_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: types.ModuleType
) -> None:
    """Newly added packages should be collected without manual configuration."""

    stub_kgfoundry = types.ModuleType("kgfoundry")
    stub_kgfoundry.__all__ = ("kgfoundry_common",)
    monkeypatch.setitem(sys.modules, "kgfoundry", stub_kgfoundry)

    src_root = tmp_path / "src"
    for package_name in ("kgfoundry_common", "new_package"):
        package_root = src_root / package_name
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(module, "SRC_ROOT", src_root)

    loaded_packages: list[str] = []

    def fake_load(package: str, *args: object, **kwargs: object) -> object:
        loaded_packages.append(package)
        module_path = Path("/virtual") / Path(package.replace(".", "/")).with_suffix(".py")
        return _make_stub_module(package, module_path)

    monkeypatch.setattr(module, "_load", fake_load)

    module._get_package_roots.cache_clear()
    modules: dict[str, object] | None = None
    try:
        packages = module._get_package_roots()
        assert "new_package" in packages

        module._get_package_roots.cache_clear()
        modules = module._collect_modules(extensions_bundle=object())
    finally:
        module._get_package_roots.cache_clear()

    assert "new_package" in loaded_packages
    assert modules is not None and "new_package" in modules
