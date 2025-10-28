"""Tests for tools.auto_docstrings helpers."""

from __future__ import annotations

import ast
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
