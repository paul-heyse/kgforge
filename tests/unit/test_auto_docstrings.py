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
from importlib import util
from pathlib import Path
import sys

import pytest

AUTO_DOCSTRINGS_PATH = Path(__file__).resolve().parents[2] / "tools" / "auto_docstrings.py"
_SPEC = util.spec_from_file_location("tools.auto_docstrings", AUTO_DOCSTRINGS_PATH)
assert _SPEC and _SPEC.loader  # pragma: no cover - loading guard
auto_docstrings = util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = auto_docstrings
_SPEC.loader.exec_module(auto_docstrings)


@pytest.mark.parametrize(
    "source, module_name, expected_lines",
    [
        (
            """
async def async_example(value: int) -> str:
    return str(value)
""",
            "pkg.sample",
            [
                ">>> from pkg.sample import async_example",
                ">>> result = async_example(...)",
                ">>> result  # doctest: +ELLIPSIS",
                "...",
            ],
        ),
        (
            """
def sync_example() -> None:
    pass
""",
            "pkg.sample",
            [
                ">>> sync_example()  # doctest: +ELLIPSIS",
            ],
        ),
    ],
)
def test_build_docstring_appends_examples(
    source: str, module_name: str, expected_lines: list[str]
) -> None:
    """Ensure Examples block is appended for functions and async functions."""

    node = ast.parse(source).body[0]
    doc_lines = auto_docstrings.build_docstring("function", node, module_name)

    assert "Examples" in doc_lines
    examples_index = doc_lines.index("Examples")
    assert doc_lines[examples_index + 1] == "--------"

    closing_index = len(doc_lines) - 1
    emitted_examples = doc_lines[examples_index + 2 : closing_index]
    for line in expected_lines:
        assert line in emitted_examples
