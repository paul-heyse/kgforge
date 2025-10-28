"""Tests for ``tools.add_module_docstrings``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import add_module_docstrings


@pytest.fixture
def fake_src(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary ``src`` tree and patch the module to use it."""

    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setattr(add_module_docstrings, "SRC", src)
    return src


def test_module_name_for_regular_modules(fake_src: Path) -> None:
    """Regular Python modules should preserve their dotted path."""

    module_path = fake_src / "pkg" / "module.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("pass\n")

    assert add_module_docstrings.module_name(module_path) == "pkg.module"


def test_module_name_strips_init_suffix(fake_src: Path) -> None:
    """Package ``__init__`` files should map to the containing package name."""

    init_path = fake_src / "pkg" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    init_path.write_text("pass\n")

    assert add_module_docstrings.module_name(init_path) == "pkg"


def test_module_name_for_nested_packages(fake_src: Path) -> None:
    """Nested package ``__init__`` files should flatten to their package path."""

    nested_init = fake_src / "pkg" / "subpkg" / "__init__.py"
    nested_init.parent.mkdir(parents=True)
    nested_init.write_text("pass\n")

    assert add_module_docstrings.module_name(nested_init) == "pkg.subpkg"


def test_insert_docstring_uses_package_name_for_init(fake_src: Path) -> None:
    """Docstrings inserted for packages should reference the clean package name."""

    init_path = fake_src / "pkg" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    init_path.write_text("pass\n")

    assert add_module_docstrings.insert_docstring(init_path) is True
    assert init_path.read_text().startswith('"""Module for pkg."""\n\npass')

