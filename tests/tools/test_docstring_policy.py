from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path

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
) -> SemanticResult:
    symbol = SymbolHarvest(
        qname=qname,
        module="pkg.module",
        kind="function",
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
        decorators=[],
        is_async=False,
        is_generator=False,
    )
    schema = DocstringSchema(
        summary=summary,
        parameters=[
            ParameterDoc(
                name="value",
                annotation="int",
                description=parameter_description,
                optional=False,
                default=None,
                display_name="value",
                kind="positional_or_keyword",
            )
        ],
        returns=[ReturnDoc(annotation="int", description=return_description, kind="returns")],
    )
    return SemanticResult(symbol=symbol, schema=schema)


def test_policy_reports_missing_descriptions(tmp_path: Path) -> None:
    engine = PolicyEngine(PolicySettings())
    engine.record([
        _make_semantic(
            tmp_path,
            summary="TODO fill summary",
            parameter_description="todo",
            return_description="todo",
        )
    ])
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

