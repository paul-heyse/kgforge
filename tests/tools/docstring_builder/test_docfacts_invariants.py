"""Regression tests for DocFacts invariants enforced during artifact builds."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.docstring_builder.docfacts import build_docfacts
from tools.docstring_builder.models import (
    SchemaViolationError,
    validate_docfacts_payload,
)
from tools.docstring_builder.schema import DocstringSchema
from tools.docstring_builder.semantics import SemanticResult


def _semantic_result_with_lines(lineno: int, end_lineno: int | None) -> SemanticResult:
    symbol = SimpleNamespace(
        qname="pkg.module.func",
        module="pkg.module",
        kind="function",
        parameters=[],
        return_annotation=None,
        docstring=None,
        owned=True,
        filepath=Path("src/pkg/module.py"),
        lineno=lineno,
        end_lineno=end_lineno,
        col_offset=0,
        decorators=[],
        is_async=False,
        is_generator=False,
    )
    schema = DocstringSchema(summary="Describe func.")
    return SemanticResult(symbol=symbol, schema=schema)


def test_build_docfacts_normalizes_invalid_line_numbers() -> None:
    result = _semantic_result_with_lines(lineno=0, end_lineno=0)

    docfacts = build_docfacts([result])

    assert docfacts[0].lineno == 1
    assert docfacts[0].end_lineno is None


def test_validate_docfacts_payload_surfaces_schema_violation_details() -> None:
    payload = {
        "docfactsVersion": "2.0",
        "provenance": {
            "builderVersion": "1.0.0",
            "configHash": "0" * 64,
            "commitHash": "deadbeef",
            "generatedAt": "2025-01-01T00:00:00Z",
        },
        "entries": [
            {
                "qname": "pkg.module.func",
                "module": "pkg.module",
                "kind": "function",
                "filepath": "src/pkg/module.py",
                "lineno": 0,
                "end_lineno": None,
                "decorators": [],
                "is_async": False,
                "is_generator": False,
                "owned": True,
                "parameters": [],
                "returns": [],
                "raises": [],
                "notes": [],
            }
        ],
    }

    with pytest.raises(SchemaViolationError) as excinfo:
        validate_docfacts_payload(payload)

    problem = excinfo.value.problem
    assert problem is not None
    pointer = problem.get("extensions", {}).get("jsonPointer")
    assert isinstance(pointer, str)
    assert pointer.endswith("/lineno")
