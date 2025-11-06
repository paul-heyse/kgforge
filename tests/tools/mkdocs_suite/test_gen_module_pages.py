import importlib
import io
import logging
import sys
import types

from pathlib import Path

import pytest


def _install_mkdocs_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight ``mkdocs_gen_files`` stub for module imports."""

    def _fake_open(path: object, *_args: object, **_kwargs: object) -> io.StringIO:
        return io.StringIO()

    stub = types.ModuleType("mkdocs_gen_files")
    setattr(stub, "open", _fake_open)  # noqa: B010
    setattr(stub, "files", io.StringIO)  # noqa: B010
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", stub)


def _install_griffe_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stub ``griffe`` module that forces module discovery to degrade."""

    stub = types.ModuleType("griffe")

    class _StubGriffeError(Exception):
        """Replacement for :class:`griffe.GriffeError` used during testing."""

    def _stub_load(*_args: object, **_kwargs: object) -> object:
        raise _StubGriffeError("stub discovery failure")

    def _stub_load_extensions(*_args: object, **_kwargs: object) -> object:
        return object()

    setattr(stub, "GriffeError", _StubGriffeError)
    setattr(stub, "load", _stub_load)
    setattr(stub, "load_extensions", _stub_load_extensions)
    monkeypatch.setitem(sys.modules, "griffe", stub)


def test_render_module_pages_warns_and_continues_on_invalid_api_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid API usage JSON should be ignored with a warning during builds."""

    _install_mkdocs_stub(monkeypatch)
    _install_griffe_stub(monkeypatch)

    # Prevent registry lookups from touching the filesystem during import.
    import tools._shared.augment_registry as augment_registry

    monkeypatch.setattr(augment_registry, "load_registry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(augment_registry, "render_problem_details", lambda _exc: {})

    module_name = "tools.mkdocs_suite.docs._scripts.gen_module_pages"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    caplog.clear()
    caplog.set_level(logging.WARNING, logger=module.LOGGER.name)

    invalid_file = tmp_path / "api_usage.json"
    invalid_file.write_text("{not-json", encoding="utf-8")

    monkeypatch.setattr(module, "API_USAGE_FILE", invalid_file)
    monkeypatch.setattr(module, "REGISTRY_PATH", tmp_path / "api_registry.yaml")

    discovered: dict[str, bool] = {}

    def _stub_discover_extensions(*_args: object, **_kwargs: object) -> tuple[list[str], None]:
        discovered["called"] = True
        return [], None

    monkeypatch.setattr(module, "_discover_extensions", _stub_discover_extensions)
    monkeypatch.setattr(module, "_collect_modules", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "_render_module_page", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_nav_metadata_for_module", lambda *_args, **_kwargs: {})

    module.render_module_pages()

    assert discovered.get("called") is True
    assert any("Skipping API usage map" in record.message for record in caplog.records)
