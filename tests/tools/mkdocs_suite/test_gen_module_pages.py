"""Tests for the MkDocs module page generator helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import io
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(name="gen_module_pages")
def fixture_gen_module_pages(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load ``gen_module_pages`` without executing its expensive side effects."""
    mkdocs_stub = types.ModuleType("mkdocs_gen_files")
    mkdocs_stub.open = lambda *_args, **_kwargs: io.StringIO()
    mkdocs_stub.files = io.StringIO
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", mkdocs_stub)

    tools_stub = types.ModuleType("tools")
    tools_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", tools_stub)

    shared_stub = types.ModuleType("tools._shared")
    shared_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools._shared", shared_stub)

    augment_stub = types.ModuleType("tools._shared.augment_registry")
    augment_stub.AugmentRegistryError = type("AugmentRegistryError", (Exception,), {})
    augment_stub.load_registry = lambda *_args, **_kwargs: None
    augment_stub.render_problem_details = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "tools._shared.augment_registry", augment_stub)

    mkdocs_suite_stub = types.ModuleType("tools.mkdocs_suite")
    mkdocs_suite_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite", mkdocs_suite_stub)

    docs_stub = types.ModuleType("tools.mkdocs_suite.docs")
    docs_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite.docs", docs_stub)

    scripts_stub = types.ModuleType("tools.mkdocs_suite.docs._scripts")
    scripts_stub.__path__ = []  # type: ignore[attr-defined]
    scripts_stub.load_repo_settings = lambda *_args, **_kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite.docs._scripts", scripts_stub)

    msgspec_stub = types.ModuleType("msgspec")
    msgspec_stub.UNSET = object()

    class _Struct:
        """Minimal stand-in for :class:`msgspec.Struct`."""

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__()

    msgspec_stub.Struct = _Struct
    msgspec_stub.structs = types.SimpleNamespace(replace=lambda obj, **_kwargs: obj)
    msgspec_stub.field = lambda *_args, **kwargs: kwargs
    msgspec_stub.json = types.SimpleNamespace(
        encode=lambda *_args, **_kwargs: b"", decode=lambda *_args, **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "msgspec", msgspec_stub)

    problem_details_stub = types.SimpleNamespace(ProblemDetailsDict=dict)
    typing_stub = types.ModuleType("kgfoundry_common.typing")
    typing_stub.gate_import = lambda *_args, **_kwargs: problem_details_stub
    kgfoundry_common_stub = types.ModuleType("kgfoundry_common")
    kgfoundry_common_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kgfoundry_common", kgfoundry_common_stub)
    monkeypatch.setitem(sys.modules, "kgfoundry_common.typing", typing_stub)
    errors_stub = types.ModuleType("kgfoundry_common.errors")
    errors_stub.SchemaValidationError = type("SchemaValidationError", (Exception,), {})
    serialization_stub = types.ModuleType("kgfoundry_common.serialization")
    serialization_stub.validate_payload = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "kgfoundry_common.errors", errors_stub)
    monkeypatch.setitem(sys.modules, "kgfoundry_common.serialization", serialization_stub)

    logging_stub = types.ModuleType("kgfoundry_common.logging")

    class _LoggerAdapter:
        def __init__(
            self, logger: object | None = None, extra: dict[str, object] | None = None
        ) -> None:
            self.logger = logger or types.SimpleNamespace()
            self.extra = extra or {}

    logging_stub.LoggerAdapter = _LoggerAdapter
    logging_stub.get_logger = lambda *_args, **_kwargs: _LoggerAdapter()
    monkeypatch.setitem(sys.modules, "kgfoundry_common.logging", logging_stub)

    griffe_stub = types.ModuleType("griffe")
    griffe_stub.load = lambda *_args, **_kwargs: object()
    griffe_stub.load_extensions = lambda *_args, **_kwargs: None
    griffe_stub.GriffeError = Exception
    monkeypatch.setitem(sys.modules, "griffe", griffe_stub)

    original_spec = importlib.util.spec_from_file_location

    class _StubLoader(importlib.machinery.SourceFileLoader):
        def __init__(self, name: str, initializer: Callable[[types.ModuleType], None]) -> None:
            super().__init__(name, "<stub>")
            self._initializer = initializer

        def exec_module(self, module: types.ModuleType) -> None:
            """Populate the stub module without reading from disk."""
            self._initializer(module)

    def _navmap_initializer(module: types.ModuleType) -> None:
        module.DEFAULT_EXTENSIONS = []
        module.DEFAULT_SEARCH_PATHS = []

    def _nav_loader_initializer(module: types.ModuleType) -> None:
        module.load_nav_metadata = lambda *_args, **_kwargs: {}

    def _fake_spec_from_file_location(
        name: str, location: str | Path, *args: object, **kwargs: object
    ) -> importlib.machinery.ModuleSpec:
        path_name = Path(location).name
        if path_name == "griffe_navmap.py":
            loader = _StubLoader(name, _navmap_initializer)
            return importlib.machinery.ModuleSpec(name, loader)
        if path_name == "navmap_loader.py":
            loader = _StubLoader(name, _nav_loader_initializer)
            return importlib.machinery.ModuleSpec(name, loader)
        return original_spec(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", _fake_spec_from_file_location)

    module_path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "mkdocs_suite"
        / "docs"
        / "_scripts"
        / "gen_module_pages.py"
    )
    module_code = module_path.read_text(encoding="utf-8")
    module_ast = ast.parse(module_code, filename=str(module_path))
    if module_ast.body and isinstance(module_ast.body[-1], ast.Expr):
        call = module_ast.body[-1].value
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "render_module_pages":
            module_ast.body.pop()

    compiled = compile(module_ast, str(module_path), "exec")
    module = types.ModuleType("gen_module_pages_under_test")
    module.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compiled, module.__dict__)
    return module


def test_inline_d2_neighborhood_uses_relative_doc_paths(gen_module_pages: types.ModuleType) -> None:
    """D2 neighborhood links should point to relative module documentation paths."""
    module_path = "kgfoundry_common.logging"
    ModuleFacts = gen_module_pages.ModuleFacts
    facts = ModuleFacts(
        imports={module_path: {"kgfoundry_common.config"}},
        imported_by={module_path: {"kgfoundry_common.app"}},
        exports={},
        classes={},
        functions={},
        bases={},
        api_usage={},
        documented_modules={module_path, "kgfoundry_common.config", "kgfoundry_common.app"},
        source_paths={},
    )

    block = gen_module_pages._inline_d2_neighborhood(module_path, facts)

    assert 'link: "./kgfoundry_common/logging.md"' in block
    assert 'link: "./kgfoundry_common/config.md"' in block
    assert 'link: "./kgfoundry_common/app.md"' in block


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

    stub.GriffeError = _StubGriffeError
    stub.load = _stub_load
    stub.load_extensions = _stub_load_extensions
    monkeypatch.setitem(sys.modules, "griffe", stub)


def test_render_module_pages_warns_and_continues_on_invalid_api_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid API usage JSON should be ignored with a warning during builds."""
    _install_mkdocs_stub(monkeypatch)
    _install_griffe_stub(monkeypatch)

    # Prevent registry lookups from touching the filesystem during import.
    from tools._shared import augment_registry

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
