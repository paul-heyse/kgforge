"""Regression tests for MkDocs interface catalog generation."""

from __future__ import annotations

import importlib
import io
import json
import logging
import sys
import types
from typing import TYPE_CHECKING, Self

import pytest

if TYPE_CHECKING:
    from pathlib import Path

MODULE_PATH = "tools.mkdocs_suite.docs._scripts.gen_interface_pages"


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
    module = importlib.import_module(MODULE_PATH)
    monkeypatch.setattr(module, "REPO_ROOT", temporary_repo)

    interfaces = module.collect_nav_interfaces()

    assert any("invalid/_nav.json" in record.message for record in caplog.records)
    assert interfaces == [{"id": "valid-interface", "module": "valid", "_nav_module_path": "valid"}]


class _DummyFile(io.StringIO):
    """Collect writes made through ``mkdocs_gen_files`` stubs."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def __enter__(self) -> Self:
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

        """
        _CAPTURED_OUTPUTS[self._path] = self.getvalue()


_CAPTURED_OUTPUTS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _clear_captured_outputs() -> None:
    _CAPTURED_OUTPUTS.clear()


def _install_mkdocs_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a lightweight ``mkdocs_gen_files`` shim for testing."""

    def _fake_open(path: object, *_args: object, **_kwargs: object) -> _DummyFile:
        return _DummyFile(str(path))

    stub = types.ModuleType("mkdocs_gen_files")
    stub.open = _fake_open
    stub.files = io.StringIO
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
    assert "pkg_root.subpkg.leaf" in output

    sys.modules.pop(module_name, None)


def test_write_interface_table_escapes_markdown_control_characters() -> None:
    """User-controlled metadata should be escaped before inserting into tables."""
    handle = io.StringIO()
    interfaces: list[dict[str, object]] = [
        {
            "id": "interface|id\nwith newline",
            "type": "service|type",
            "module": "pkg.module",
            "owner": "owner|name`tick`",
            "stability": "beta|1",
            "spec": "http://example.com/spec|path",
            "description": "line1|\nline2`",
            "problem_details": ["issue|one", "issue`two`"],
        }
    ]

    module = importlib.import_module(MODULE_PATH)
    module.write_interface_table(handle, interfaces, registry=None)

    lines = handle.getvalue().strip().splitlines()
    row = lines[-1]
    cells = [cell.strip() for cell in row.strip().strip("|").split(" | ")]

    assert len(cells) == 8
    assert cells[0] == "interface\\|id<br />with newline"
    assert cells[1] == "service\\|type"
    assert cells[2] == "pkg.module"
    assert cells[3] == "owner\\|name\\`tick\\`"
    assert cells[4] == "beta\\|1"
    assert cells[5] == "http://example.com/spec\\|path"
    assert cells[6] == "line1\\|<br />line2\\`"
    assert cells[7] == "issue\\|one, issue\\`two\\`"


def test_operation_href_returns_relative_anchor() -> None:
    """Operation links should include encoded anchors relative to the doc path."""
    module = importlib.import_module(MODULE_PATH)

    href = module._operation_href("docs/api/openapi-cli.yaml", "cli.operation/with space")

    assert href == "openapi-cli.md#operation/cli.operation%2Fwith%20space"
