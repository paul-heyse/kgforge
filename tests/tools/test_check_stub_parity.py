"""Tests for the stub parity checker CLI."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict, cast

import pytest
import tools.check_stub_parity as stub_parity

from kgfoundry_common.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


class StubParityIssue(TypedDict):
    missing_symbols: list[str]


class StubParityContext(TypedDict):
    issue_count: int
    error_count: int
    issues: list[StubParityIssue]


def _register_runtime_module(name: str, exports: tuple[str, ...]) -> None:
    module = ModuleType(name)
    module.__all__ = list(exports)

    def _placeholder(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        return None

    for export in exports:
        setattr(module, export, _placeholder)

    sys.modules[name] = module


def test_evaluate_stub_reports_any_usage(tmp_path: Path) -> None:
    """`evaluate_stub` should capture Any usages for downstream reporting."""
    module_name = "stub_parity_any_usage"
    _register_runtime_module(module_name, ("api",))

    stub_path = tmp_path / f"{module_name}.pyi"
    stub_path.write_text(
        "from typing import Any\n\ndef api(value: Any) -> None: ...\n",
        encoding="utf-8",
    )

    outcome = stub_parity.evaluate_stub(module_name, stub_path)

    assert outcome.any_usages
    assert outcome.any_usages[0][0] == 3  # First element is line_number
    assert outcome.has_errors


def test_run_stub_parity_checks_success(tmp_path: Path) -> None:
    """A fully matching stub should complete without raising."""
    module_name = "stub_parity_success"
    _register_runtime_module(module_name, ("foo",))

    stub_path = tmp_path / f"{module_name}.pyi"
    stub_path.write_text("def foo() -> None: ...\n", encoding="utf-8")

    stub_parity.run_stub_parity_checks([(module_name, stub_path)])


def test_run_stub_parity_checks_raises_with_context(tmp_path: Path) -> None:
    """Missing runtime exports should surface through `ConfigurationError` context."""
    module_name = "stub_parity_missing"
    _register_runtime_module(module_name, ("foo",))

    stub_path = tmp_path / f"{module_name}.pyi"
    stub_path.write_text("def bar() -> None: ...\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        stub_parity.run_stub_parity_checks([(module_name, stub_path)])

    context = excinfo.value.context
    assert context is not None
    context_dict = cast("StubParityContext", context)
    assert context_dict["issue_count"] == 1
    assert context_dict["error_count"] == 1
    issues = context_dict["issues"]
    issue = issues[0]
    assert issue["missing_symbols"] == ["foo"]
