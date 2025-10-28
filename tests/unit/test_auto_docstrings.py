"""Tests for ``tools.auto_docstrings`` helpers."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "auto_docstrings.py"
spec = importlib.util.spec_from_file_location("auto_docstrings", MODULE_PATH)
auto_docstrings = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auto_docstrings
assert spec.loader is not None
spec.loader.exec_module(auto_docstrings)

_required_sections = auto_docstrings._required_sections
annotation_to_text = auto_docstrings.annotation_to_text
build_docstring = auto_docstrings.build_docstring
build_examples = auto_docstrings.build_examples
parameters_for = auto_docstrings.parameters_for


def _get_function(code: str) -> ast.FunctionDef:
    module = ast.parse(code)
    node = module.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_build_docstring_appends_examples_for_public_function() -> None:
    node = _get_function(
        """
def do_stuff(value: int) -> str:
    return str(value)
"""
    )

    lines = build_docstring("function", node, "pkg.module")

    params = parameters_for(node)
    expected_tail = ["", *build_examples("pkg.module", "do_stuff", params, True)]

    assert lines[-(len(expected_tail) + 1) : -1] == expected_tail
    assert lines[-1] == '"""'


def test_build_docstring_skips_examples_for_private_function() -> None:
    node = _get_function(
        """
def _hidden(value: int) -> None:
    return None
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    assert "Examples" not in docstring


def test_required_sections_satisfied_by_generated_docstring() -> None:
    node = _get_function(
        """
def process(item: str, limit: int | None = None) -> str:
    return item
"""
    )

    lines = build_docstring("function", node, "pkg.module")
    docstring = "\n".join(lines)

    params = parameters_for(node)
    returns = annotation_to_text(node.returns)
    required = _required_sections("function", params, returns, [])

    for section in required:
        assert section in docstring
