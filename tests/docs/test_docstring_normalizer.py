"""Tests for the docstring normalisation helpers."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from tools.docstring_builder.harvest import ParameterHarvest, SymbolHarvest
from tools.docstring_builder.normalizer import normalize_docstring


def sample_function(texts: list[str], filters: dict[str, int] | None = None) -> dict[str, int]:
    """Do something useful.

    Parameters
    ----------
    texts : List[str]
        Example payload.
    filters : Dict[str, int]
        Filtering rules.

    Returns
    -------
    Dict[str, int]
        Response payload.
    """
    del filters
    return {"example": len(texts)}


class SampleClass:
    """Simple container for testing."""

    def method(self, values: list[int]) -> list[int]:
        """Double values.

        Parameters
        ----------
        values : List[int]
            Values to process. Demonstrating list[str] normalization.

        Returns
        -------
        List[int]
            Processed values.
        """
        return [value * 2 for value in values]


@pytest.mark.parametrize(
    "symbol",
    [
        SymbolHarvest(
            qname="tests.docs.test_docstring_normalizer.sample_function",
            module="tests.docs.test_docstring_normalizer",
            kind="function",
            parameters=[
                ParameterHarvest(
                    name="texts",
                    kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation="list[str]",
                    default=None,
                ),
                ParameterHarvest(
                    name="filters",
                    kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation="dict[str, int] | None",
                    default="None",
                ),
            ],
            return_annotation="dict[str, int]",
            docstring=sample_function.__doc__,
            owned=True,
            filepath=Path(__file__),
            lineno=sample_function.__code__.co_firstlineno,
            end_lineno=None,
            col_offset=0,
            decorators=[],
            is_async=False,
            is_generator=False,
        ),
        SymbolHarvest(
            qname="tests.docs.test_docstring_normalizer.SampleClass.method",
            module="tests.docs.test_docstring_normalizer",
            kind="method",
            parameters=[
                ParameterHarvest(
                    name="self",
                    kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation=None,
                    default=None,
                ),
                ParameterHarvest(
                    name="values",
                    kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation="list[int]",
                    default=None,
                ),
            ],
            return_annotation="list[int]",
            docstring=SampleClass.method.__doc__,
            owned=True,
            filepath=Path(__file__),
            lineno=SampleClass.method.__code__.co_firstlineno,
            end_lineno=None,
            col_offset=0,
            decorators=[],
            is_async=False,
            is_generator=False,
        ),
    ],
)
def test_normalize_docstring_round_trip(symbol: SymbolHarvest) -> None:
    """Docstring normalisation mirrors annotations and preserves descriptions."""
    marker = "<!-- auto:docstring-builder v1 -->"
    updated = normalize_docstring(symbol, marker)
    assert updated is not None
    assert marker in updated
    assert "list[str]" in updated
    assert "dict[str, int]" in updated or "list[int]" in updated
    assert "Example payload." in updated or "Values to process." in updated
    assert "optional" in updated or symbol.kind == "method"
