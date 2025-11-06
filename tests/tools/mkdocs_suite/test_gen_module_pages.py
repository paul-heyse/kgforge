"""Tests for the MkDocs module page generator helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import importlib.util
import io
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import griffe
import pytest
from tools._shared import augment_registry


@pytest.fixture(name="gen_module_pages")
def fixture_gen_module_pages(  # noqa: C901, PLR0914, PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    """Load ``gen_module_pages`` without executing its expensive side effects.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture for stubbing module imports.

    Returns
    -------
    types.ModuleType
        The loaded module instance with side effects stubbed out.

    Raises
    ------
    RuntimeError
        Raised when unable to construct module spec for gen_module_pages.
    """

    def _module_stub(name: str) -> Any:
        return cast("Any", types.ModuleType(name))

    mkdocs_stub = cast("Any", types.ModuleType("mkdocs_gen_files"))
    mkdocs_stub.open = lambda *_args, **_kwargs: io.StringIO()
    mkdocs_stub.files = io.StringIO
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", mkdocs_stub)

    tools_stub = cast("Any", types.ModuleType("tools"))
    tools_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "tools", tools_stub)

    shared_stub = cast("Any", types.ModuleType("tools._shared"))
    shared_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "tools._shared", shared_stub)

    augment_stub = cast("Any", types.ModuleType("tools._shared.augment_registry"))
    augment_stub.AugmentRegistryError = type("AugmentRegistryError", (Exception,), {})
    augment_stub.load_registry = lambda *_args, **_kwargs: None
    augment_stub.render_problem_details = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "tools._shared.augment_registry", augment_stub)

    operation_links_stub = cast(
        "Any", types.ModuleType("tools.mkdocs_suite.docs._scripts._operation_links")
    )
    operation_links_stub.build_operation_href = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "tools.mkdocs_suite.docs._scripts._operation_links", operation_links_stub
    )

    mkdocs_suite_stub = cast("Any", types.ModuleType("tools.mkdocs_suite"))
    mkdocs_suite_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite", mkdocs_suite_stub)

    docs_stub = cast("Any", types.ModuleType("tools.mkdocs_suite.docs"))
    docs_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite.docs", docs_stub)

    scripts_stub = cast("Any", types.ModuleType("tools.mkdocs_suite.docs._scripts"))
    scripts_stub.__path__ = []
    scripts_stub.load_repo_settings = lambda *_args, **_kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite.docs._scripts", scripts_stub)

    msgspec_stub = cast("Any", types.ModuleType("msgspec"))
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
    typing_stub = cast("Any", types.ModuleType("kgfoundry_common.typing"))
    typing_stub.gate_import = lambda *_args, **_kwargs: problem_details_stub
    kgfoundry_common_stub = cast("Any", types.ModuleType("kgfoundry_common"))
    kgfoundry_common_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "kgfoundry_common", kgfoundry_common_stub)
    monkeypatch.setitem(sys.modules, "kgfoundry_common.typing", typing_stub)
    errors_stub = cast("Any", types.ModuleType("kgfoundry_common.errors"))
    errors_stub.SchemaValidationError = type("SchemaValidationError", (Exception,), {})
    serialization_stub = cast("Any", types.ModuleType("kgfoundry_common.serialization"))
    serialization_stub.validate_payload = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "kgfoundry_common.errors", errors_stub)
    monkeypatch.setitem(sys.modules, "kgfoundry_common.serialization", serialization_stub)

    logging_stub = cast("Any", types.ModuleType("kgfoundry_common.logging"))

    class _LoggerAdapter:
        def __init__(
            self, logger: object | None = None, extra: dict[str, object] | None = None
        ) -> None:
            self.logger = logger or types.SimpleNamespace()
            self.extra = extra or {}

    logging_stub.LoggerAdapter = _LoggerAdapter
    logging_stub.get_logger = lambda *_args, **_kwargs: _LoggerAdapter()
    monkeypatch.setitem(sys.modules, "kgfoundry_common.logging", logging_stub)

    original_spec = importlib.util.spec_from_file_location

    class _StubLoader(importlib.machinery.SourceFileLoader):
        def __init__(self, name: str, initializer: Callable[[types.ModuleType], None]) -> None:
            super().__init__(name, "<stub>")
            self._initializer = initializer

        def exec_module(self, module: types.ModuleType) -> None:
            """Populate the stub module without reading from disk."""
            self._initializer(module)

    def _navmap_initializer(module: types.ModuleType) -> None:
        module_any = cast("Any", module)
        module_any.DEFAULT_EXTENSIONS = []
        module_any.DEFAULT_SEARCH_PATHS = []

    def _nav_loader_initializer(module: types.ModuleType) -> None:
        module_any = cast("Any", module)
        module_any.load_nav_metadata = lambda *_args, **_kwargs: {}

        class _NavMetadataModel(dict):
            def as_mapping(self) -> dict[str, object]:
                return dict(self)

        module_any.NavMetadataModel = _NavMetadataModel

    def _fake_spec_from_file_location(
        name: str, location: str | Path, *args: Any, **kwargs: Any
    ) -> importlib.machinery.ModuleSpec:
        path_name = Path(location).name
        if path_name == "griffe_navmap.py":
            loader = _StubLoader(name, _navmap_initializer)
            return importlib.machinery.ModuleSpec(name, loader)
        if path_name == "navmap_loader.py":
            loader = _StubLoader(name, _nav_loader_initializer)
            return importlib.machinery.ModuleSpec(name, loader)
        spec = original_spec(name, location, *args, **kwargs)
        if spec is None or spec.loader is None:
            message = "Unable to construct module spec"
            raise RuntimeError(message)
        return spec

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
    if module_ast.body:
        last_stmt = module_ast.body[-1]
        if isinstance(last_stmt, ast.Expr) and isinstance(last_stmt.value, ast.Call):
            call = last_stmt.value
            if getattr(call.func, "id", None) == "render_module_pages":
                module_ast.body.pop()

    compiled = compile(module_ast, str(module_path), "exec")

    class _PatchedLoader(importlib.machinery.SourceFileLoader):
        def __init__(self, fullname: str, path: str, code_obj: types.CodeType) -> None:
            super().__init__(fullname, path)
            self._code_obj = code_obj

        def get_code(self, fullname: str) -> types.CodeType:
            del fullname
            return self._code_obj

    module_name = "gen_module_pages_under_test"
    loader = _PatchedLoader(module_name, str(module_path), compiled)
    spec = importlib.util.spec_from_file_location(module_name, module_path, loader=loader)
    if spec is None or spec.loader is None:
        message = "Unable to construct module spec for gen_module_pages"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
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

    block = gen_module_pages._inline_d2_neighborhood(module_path, facts)  # noqa: SLF001

    assert 'link: "./kgfoundry_common/logging.md"' in block
    assert 'link: "./kgfoundry_common/config.md"' in block
    assert 'link: "./kgfoundry_common/app.md"' in block


def _install_mkdocs_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight ``mkdocs_gen_files`` stub for module imports."""

    def _fake_open(path: object, *_args: object, **_kwargs: object) -> io.StringIO:
        del path
        return io.StringIO()

    stub = types.ModuleType("mkdocs_gen_files")
    setattr(stub, "open", _fake_open)  # noqa: B010
    setattr(stub, "files", io.StringIO)  # noqa: B010
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", stub)


def test_render_module_pages_warns_and_continues_on_invalid_api_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid API usage JSON should be ignored with a warning during builds."""
    _install_mkdocs_stub(monkeypatch)

    message = "stub discovery failure"

    def _raise_loader_error(*_args: object, **_kwargs: object) -> object:
        raise griffe.GriffeError(message)

    monkeypatch.setattr(griffe, "load", _raise_loader_error)
    monkeypatch.setattr(griffe, "load_extensions", lambda *_args, **_kwargs: [])

    # Prevent registry lookups from touching the filesystem during import.
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
