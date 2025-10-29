from __future__ import annotations

import importlib

import pytest

pytest.importorskip("docstring_parser")

import sitecustomize  # noqa: F401


def test_docstring_parser_common_surface() -> None:
    common = importlib.import_module("docstring_parser.common")
    assert hasattr(common, "Docstring")
    assert hasattr(common, "DocstringReturns")
    assert hasattr(common, "DocstringYields")

    returns = common.DocstringReturns(
        args=["value"],
        description="desc",
        type_name="str",
        is_generator=False,
        return_name="value",
    )
    doc = common.Docstring()
    doc.meta.append(returns)
    assert doc.meta[-1] is returns
