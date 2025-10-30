import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


def _reload_sitecustomize(monkeypatch: pytest.MonkeyPatch, flag: str | None = None) -> ModuleType:
    if flag is None:
        monkeypatch.delenv("KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE", raising=False)
    else:
        monkeypatch.setenv("KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE", flag)
    sys.modules.pop("sitecustomize", None)
    module = importlib.import_module("sitecustomize")
    return importlib.reload(module)


@pytest.mark.importorskip("docstring_parser")
def test_sitecustomize_emits_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    module = _reload_sitecustomize(monkeypatch, "1")
    warning = recwarn.pop(DeprecationWarning)
    assert "deprecated" in str(warning.message)
    assert module.ENABLE_SITECUSTOMIZE is True


def test_sitecustomize_disabled_flag(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    module = _reload_sitecustomize(monkeypatch, "0")
    assert module.ENABLE_SITECUSTOMIZE is False
    assert not recwarn


def test_sitecustomize_handles_missing_docstring_parser(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple | list = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("docstring_parser"):
            message = "docstring_parser unavailable"
            raise ModuleNotFoundError(message)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = _reload_sitecustomize(monkeypatch, "1")
    assert module.doc_common is None
    assert module.ENABLE_SITECUSTOMIZE is True
    assert not recwarn
