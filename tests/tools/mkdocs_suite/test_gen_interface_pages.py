"""Regression tests for MkDocs interface catalog generation."""

from __future__ import annotations

import importlib
import io
import json
import logging
import sys
import types
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tools.mkdocs_suite.docs._scripts import gen_interface_pages  # noqa: PLC2701


@pytest.fixture(name="temporary_repo")
def fixture_temporary_repo(tmp_path: Path) -> Path:
    """Create a temporary repository layout for nav discovery tests.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture providing a base path for test files.

    Returns
    -------
    Path
        Path to the temporary repository root directory containing test
        navigation files.
    """
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

    interfaces = gen_interface_pages._collect_nav_interfaces()  # noqa: SLF001

    assert any("invalid/_nav.json" in record.message for record in caplog.records)
    assert interfaces == [{"id": "valid-interface", "module": "valid", "_nav_module_path": "valid"}]


class _DummyFile(io.StringIO):
    """Collect writes made through ``mkdocs_gen_files`` stubs."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def __enter__(self) -> _DummyFile:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Store written content keyed by the path before closing.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type if an exception occurred, None otherwise.
        exc_val : BaseException | None
            Exception value if an exception occurred, None otherwise.
        exc_tb : types.TracebackType | None
            Traceback object if an exception occurred, None otherwise.

        Returns
        -------
        bool
            Always returns ``False`` to indicate exceptions should not
            be suppressed.
        """
        _CAPTURED_OUTPUTS[self._path] = self.getvalue()
        return None


_CAPTURED_OUTPUTS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _clear_captured_outputs() -> None:
    _CAPTURED_OUTPUTS.clear()


def _install_mkdocs_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight ``mkdocs_gen_files`` shim for testing."""

    def _fake_open(path: object, *_args: object, **_kwargs: object) -> _DummyFile:
        return _DummyFile(str(path))

    stub = types.ModuleType("mkdocs_gen_files")
    setattr(stub, "open", _fake_open)  # noqa: B010
    setattr(stub, "files", io.StringIO)  # noqa: B010
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", stub)


def test_render_interface_catalog_links_full_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
