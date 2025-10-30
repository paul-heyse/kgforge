"""Tests covering the docstring compatibility helpers in :mod:`sitecustomize`."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Docstring builder runtime shims are deprecated",
        category=DeprecationWarning,
    )
    from src import sitecustomize


@dataclass(slots=True)
class _FakeMeta(sitecustomize.DocstringMetaProto):
    description: str | None = None


@dataclass(slots=True)
class _FakeAttr(_FakeMeta, sitecustomize.DocstringAttrProto):
    args: Sequence[str] = ("attribute", "attr")


@dataclass(slots=True)
class _FakeYields(_FakeMeta, sitecustomize.DocstringYieldsProto):
    args: Sequence[str] = ()
    type_name: str | None = None
    is_generator: bool = True
    return_name: str | None = None


class _Doc:
    def __init__(self, meta: Sequence[sitecustomize.DocstringMetaProto]) -> None:
        self.meta = list(meta)
        self.short_description = "short"
        self.long_description = "long"


def test_ensure_docstring_attrs_installs_property() -> None:
    class _MutableDoc(_Doc):
        pass

    installed = sitecustomize.ensure_docstring_attrs(_MutableDoc, _FakeAttr)
    assert installed is True
    instance = _MutableDoc([_FakeAttr(description="desc")])
    instance_any = cast(Any, instance)
    assert instance_any.attrs[0].description == "desc"

    installed_again = sitecustomize.ensure_docstring_attrs(_MutableDoc, _FakeAttr)
    assert installed_again is False


def test_ensure_docstring_yields_installs_helpers() -> None:
    class _MutableDoc(_Doc):
        pass

    added_yield, added_many = sitecustomize.ensure_docstring_yields(_MutableDoc, _FakeYields)
    assert added_yield is True
    assert added_many is True

    empty_doc = _MutableDoc([])
    empty_any = cast(Any, empty_doc)
    assert empty_any.yields is None
    assert empty_any.many_yields == []

    data = _FakeYields(description="yield")
    populated = _MutableDoc([data])
    populated_any = cast(Any, populated)
    assert populated_any.yields is data
    assert populated_any.many_yields == [data]

    added_yield_again, added_many_again = sitecustomize.ensure_docstring_yields(
        _MutableDoc, _FakeYields
    )
    assert added_yield_again is False
    assert added_many_again is False


def test_ensure_docstring_size_reports_total_length() -> None:
    class _MutableDoc(_Doc):
        pass

    assert sitecustomize.ensure_docstring_size(_MutableDoc) is True
    meta = [_FakeMeta(description="meta"), _FakeMeta(description=None)]
    doc = _MutableDoc(meta)
    doc_any = cast(Any, doc)
    assert doc_any.size == len("short") + len("long") + len("meta")

    assert sitecustomize.ensure_docstring_size(_MutableDoc) is False


def test_protocol_objects_are_accessible() -> None:
    assert sitecustomize.DocstringMetaProto
    assert sitecustomize.DocstringAttrProto
    assert sitecustomize.DocstringYieldsProto

