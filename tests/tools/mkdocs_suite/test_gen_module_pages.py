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
    mkdocs_stub.open = lambda *_args, **_kwargs: io.StringIO()  # noqa: B010
    mkdocs_stub.files = io.StringIO  # noqa: B010
    monkeypatch.setitem(sys.modules, "mkdocs_gen_files", mkdocs_stub)

    tools_stub = types.ModuleType("tools")
    tools_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", tools_stub)

    shared_stub = types.ModuleType("tools._shared")
    shared_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools._shared", shared_stub)

    augment_stub = types.ModuleType("tools._shared.augment_registry")
    augment_stub.AugmentRegistryError = type("AugmentRegistryError", (Exception,), {})  # noqa: B010
    augment_stub.load_registry = lambda *_args, **_kwargs: None  # noqa: B010
    augment_stub.render_problem_details = lambda *_args, **_kwargs: {}  # noqa: B010
    monkeypatch.setitem(sys.modules, "tools._shared.augment_registry", augment_stub)

    mkdocs_suite_stub = types.ModuleType("tools.mkdocs_suite")
    mkdocs_suite_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite", mkdocs_suite_stub)

    docs_stub = types.ModuleType("tools.mkdocs_suite.docs")
    docs_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite.docs", docs_stub)

    scripts_stub = types.ModuleType("tools.mkdocs_suite.docs._scripts")
    scripts_stub.__path__ = []  # type: ignore[attr-defined]
    scripts_stub.load_repo_settings = lambda *_args, **_kwargs: (None, None)  # noqa: B010
    monkeypatch.setitem(sys.modules, "tools.mkdocs_suite.docs._scripts", scripts_stub)

    msgspec_stub = types.ModuleType("msgspec")
    msgspec_stub.UNSET = object()  # noqa: B010

    class _Struct:  # noqa: D401 - simple stub matching msgspec.Struct API surface
        """Minimal stand-in for :class:`msgspec.Struct`."""

        def __init_subclass__(cls, **kwargs: object) -> None:  # noqa: D401 - accept msgspec kwargs
            super().__init_subclass__()

    msgspec_stub.Struct = _Struct  # noqa: B010
    msgspec_stub.structs = types.SimpleNamespace(replace=lambda obj, **_kwargs: obj)  # noqa: B010
    msgspec_stub.field = lambda *_args, **kwargs: kwargs  # noqa: B010
    msgspec_stub.json = types.SimpleNamespace(
        encode=lambda *_args, **_kwargs: b"", decode=lambda *_args, **_kwargs: None
    )  # noqa: B010
    monkeypatch.setitem(sys.modules, "msgspec", msgspec_stub)

    problem_details_stub = types.SimpleNamespace(ProblemDetailsDict=dict)
    typing_stub = types.ModuleType("kgfoundry_common.typing")
    typing_stub.gate_import = lambda *_args, **_kwargs: problem_details_stub  # noqa: B010
    kgfoundry_common_stub = types.ModuleType("kgfoundry_common")
    kgfoundry_common_stub.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kgfoundry_common", kgfoundry_common_stub)
    monkeypatch.setitem(sys.modules, "kgfoundry_common.typing", typing_stub)
    errors_stub = types.ModuleType("kgfoundry_common.errors")
    errors_stub.SchemaValidationError = type("SchemaValidationError", (Exception,), {})  # noqa: B010
    serialization_stub = types.ModuleType("kgfoundry_common.serialization")
    serialization_stub.validate_payload = lambda *_args, **_kwargs: None  # noqa: B010
    monkeypatch.setitem(sys.modules, "kgfoundry_common.errors", errors_stub)
    monkeypatch.setitem(sys.modules, "kgfoundry_common.serialization", serialization_stub)

    logging_stub = types.ModuleType("kgfoundry_common.logging")

    class _LoggerAdapter:
        def __init__(
            self, logger: object | None = None, extra: dict[str, object] | None = None
        ) -> None:
            self.logger = logger or types.SimpleNamespace()
            self.extra = extra or {}

    logging_stub.LoggerAdapter = _LoggerAdapter  # noqa: B010
    logging_stub.get_logger = lambda *_args, **_kwargs: _LoggerAdapter()  # noqa: B010
    monkeypatch.setitem(sys.modules, "kgfoundry_common.logging", logging_stub)

    griffe_stub = types.ModuleType("griffe")
    griffe_stub.load = lambda *_args, **_kwargs: object()  # noqa: B010
    griffe_stub.load_extensions = lambda *_args, **_kwargs: None  # noqa: B010
    griffe_stub.GriffeError = Exception  # noqa: B010
    monkeypatch.setitem(sys.modules, "griffe", griffe_stub)

    original_spec = importlib.util.spec_from_file_location

    class _StubLoader(importlib.machinery.SourceFileLoader):
        def __init__(self, name: str, initializer: Callable[[types.ModuleType], None]) -> None:
            super().__init__(name, "<stub>")
            self._initializer = initializer

        def exec_module(self, module: types.ModuleType) -> None:  # noqa: D401
            """Populate the stub module without reading from disk."""

            self._initializer(module)

    def _navmap_initializer(module: types.ModuleType) -> None:
        module.DEFAULT_EXTENSIONS = []  # noqa: B010
        module.DEFAULT_SEARCH_PATHS = []  # noqa: B010

    def _nav_loader_initializer(module: types.ModuleType) -> None:
        module.load_nav_metadata = lambda *_args, **_kwargs: {}  # noqa: B010

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
