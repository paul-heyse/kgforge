from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path
from typing import Literal

import pytest

from tools.docstring_builder.harvest import ParameterHarvest, SymbolHarvest
from tools.docstring_builder.policy import (
    PolicyAction,
    PolicyEngine,
    PolicyException,
    PolicySettings,
)
from tools.docstring_builder.schema import DocstringSchema, ParameterDoc, ReturnDoc
from tools.docstring_builder.semantics import SemanticResult


def _make_semantic(
    tmp_path: Path,
    *,
    summary: str = "Summarise work.",
    parameter_description: str = "todo",
    return_description: str = "todo",
    qname: str = "pkg.module.func",
    kind: Literal["class", "function", "method"] = "function",
    decorators: list[str] | None = None,
    schema_parameters: list[ParameterDoc] | None = None,
    examples: list[str] | None = None,
) -> SemanticResult:
    decorators = decorators or []
    symbol = SymbolHarvest(
        qname=qname,
        module="pkg.module",
        kind=kind,
        parameters=[
            ParameterHarvest(
                name="value",
                kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="int",
                default=None,
            )
        ],
        return_annotation="int",
        docstring=None,
        owned=True,
        filepath=tmp_path / "pkg" / "module.py",
        lineno=1,
        end_lineno=2,
        col_offset=0,
        decorators=decorators,
        is_async=False,
        is_generator=False,
    )
    parameters = schema_parameters or [
        ParameterDoc(
            name="value",
            annotation="int",
            description=parameter_description,
            optional=False,
            default=None,
            display_name="value",
            kind="positional_or_keyword",
        )
    ]
    schema = DocstringSchema(
        summary=summary,
        parameters=parameters,
        returns=[ReturnDoc(annotation="int", description=return_description, kind="returns")],
        examples=examples or [],
    )
    return SemanticResult(symbol=symbol, schema=schema)


def test_policy_reports_missing_descriptions(tmp_path: Path) -> None:
    engine = PolicyEngine(PolicySettings())
    engine.record(
        [
            _make_semantic(
                tmp_path,
                summary="TODO fill summary",
                parameter_description="todo",
                return_description="todo",
            )
        ]
    )
    report = engine.finalize()
    rules = {violation.rule for violation in report.violations}
    assert {"missing-params", "missing-returns", "coverage"}.issubset(rules)
    assert report.coverage < report.threshold


def test_policy_exception_skips_missing_params(tmp_path: Path) -> None:
    exception = PolicyException(
        symbol="pkg.module.func",
        rule="missing-params",
        expires_on=dt.date.today() + dt.timedelta(days=30),
        justification="Legacy shim",
    )
    settings = PolicySettings(
        coverage_threshold=0.0,
        coverage_action=PolicyAction.WARN,
        exceptions=[exception],
    )
    engine = PolicyEngine(settings)
    engine.record(
        [
            _make_semantic(
                tmp_path,
                summary="Provide context.",
                parameter_description="todo",
                return_description="Returns the processed value.",
            )
        ]
    )
    report = engine.finalize()
    assert not any(violation.rule == "missing-params" for violation in report.violations)
    assert report.coverage >= report.threshold


def test_policy_missing_examples_respects_action(tmp_path: Path) -> None:
    settings = PolicySettings(
        coverage_threshold=0.0,
        missing_examples_action=PolicyAction.ERROR,
    )
    engine = PolicyEngine(settings)
    engine.record(
        [
            _make_semantic(
                tmp_path,
                summary="Describe value.",
                parameter_description="Explain value.",
                return_description="Explain return.",
                examples=[],
            )
        ]
    )
    report = engine.finalize()
    assert any(violation.rule == "missing-examples" for violation in report.violations)
    assert report.coverage < report.threshold


def test_policy_summary_mood_violation(tmp_path: Path) -> None:
    settings = PolicySettings(
        coverage_threshold=0.0,
        summary_mood_action=PolicyAction.ERROR,
    )
    engine = PolicyEngine(settings)
    engine.record(
        [
            _make_semantic(
                tmp_path,
                summary="Returns the cached value",
                parameter_description="Explain value.",
                return_description="Explain return.",
                examples=[">>> func()"],
            )
        ]
    )
    report = engine.finalize()
    assert any(violation.rule == "summary-mood" for violation in report.violations)


def test_policy_dataclass_parity_violation(tmp_path: Path) -> None:
    module_path = tmp_path / "pkg" / "module.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Example:\n"
            "    first: int\n"
            "    second: str\n"
        ),
        encoding="utf-8",
    )
    settings = PolicySettings(
        coverage_threshold=0.0,
        dataclass_parity_action=PolicyAction.ERROR,
    )
    engine = PolicyEngine(settings)
    parameters = [
        ParameterDoc(
            name="first",
            annotation="int",
            description="Document first.",
            optional=False,
            default=None,
            display_name="first",
            kind="positional_or_keyword",
        )
    ]
    engine.record(
        [
            _make_semantic(
                tmp_path,
                qname="pkg.module.Example",
                summary="Describe example.",
                parameter_description="",
                return_description="",
                kind="class",
                decorators=["dataclass"],
                schema_parameters=parameters,
                examples=["Example(first=1, second='two')"],
            )
        ]
    )
    report = engine.finalize()
    violation_rules = {violation.rule for violation in report.violations}
    assert "dataclass-parity" in violation_rules
