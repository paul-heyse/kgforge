"""Tests for extended summary coverage in auto-generated docstrings."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

auto_docstrings = __import__("auto_docstrings")


MAGIC_CATEGORY_EXPECTATIONS: dict[str, set[str]] = {
    "object lifecycle": {"__new__", "__del__", "__init_subclass__"},
    "attribute access": {
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__dir__",
    },
    "descriptor": {"__get__", "__set__", "__delete__", "__set_name__"},
    "pickling": {
        "__getstate__",
        "__setstate__",
        "__reduce__",
        "__reduce_ex__",
        "__getnewargs__",
        "__getnewargs_ex__",
    },
    "binary": {
        "__add__",
        "__sub__",
        "__mul__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__divmod__",
        "__pow__",
        "__lshift__",
        "__rshift__",
        "__and__",
        "__xor__",
        "__or__",
    },
    "reverse": {
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rmatmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        "__rdivmod__",
        "__rpow__",
        "__rlshift__",
        "__rrshift__",
        "__rand__",
        "__rxor__",
        "__ror__",
    },
    "in-place": {
        "__iadd__",
        "__isub__",
        "__imul__",
        "__imatmul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__ilshift__",
        "__irshift__",
        "__iand__",
        "__ixor__",
        "__ior__",
    },
    "unary": {"__neg__", "__pos__", "__abs__", "__invert__"},
    "type conversion": {
        "__int__",
        "__float__",
        "__complex__",
        "__index__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
    },
    "collection": {"__reversed__", "__length_hint__", "__missing__"},
    "type system": {"__instancecheck__", "__subclasscheck__", "__class_getitem__"},
    "misc": {
        "__bytes__",
        "__format__",
        "__sizeof__",
        "__fspath__",
        "__buffer__",
        "__release_buffer__",
    },
}


PYDANTIC_EXPECTED: set[str] = {
    "model_config",
    "model_fields",
    "model_computed_fields",
    "model_fields_set",
    "model_extra",
    "__class_vars__",
    "model_post_init",
    "model_rebuild",
    "model_parametrized_name",
    "model_dump",
    "model_dump_json",
    "model_validate",
    "model_validate_json",
    "model_copy",
    "model_construct",
    "model_serializer",
    "model_json_schema",
    "schema",
    "schema_json",
    "dict",
    "json",
    "copy",
    "__pydantic_core_schema__",
    "__pydantic_core_config__",
    "__pydantic_decorators__",
    "__pydantic_extra__",
    "__pydantic_complete__",
    "__pydantic_computed_fields__",
    "__pydantic_fields_set__",
    "__pydantic_parent_namespace__",
    "__pydantic_generic_metadata__",
    "__pydantic_model_complete__",
    "__pydantic_serializer__",
    "__pydantic_validator__",
    "__pydantic_custom_init__",
    "__pydantic_private__",
    "__private_attributes__",
    "__pydantic_root_model__",
    "__pydantic_setattr_handlers__",
    "__pydantic_init_subclass__",
    "__pydantic_post_init__",
    "__get_pydantic_core_schema__",
    "__get_pydantic_json_schema__",
    "__signature__",
}


def _is_multi_sentence(text: str) -> bool:
    sentences = [segment for segment in re.split(r"\.\s+", text.strip()) if segment]
    return len(sentences) >= 2 and text.strip().endswith(".")


def _parse_class(source: str) -> ast.ClassDef:
    module = ast.parse(source)
    node = module.body[0]
    assert isinstance(node, ast.ClassDef)
    return node


@pytest.mark.parametrize("category", sorted(MAGIC_CATEGORY_EXPECTATIONS))
def test_magic_method_categories_have_entries(category: str) -> None:
    names = MAGIC_CATEGORY_EXPECTATIONS[category]
    missing = names.difference(auto_docstrings.MAGIC_METHOD_EXTENDED_SUMMARIES)
    assert not missing, f"{category} missing {sorted(missing)}"


def test_magic_methods_cover_required_count() -> None:
    assert len(auto_docstrings.MAGIC_METHOD_EXTENDED_SUMMARIES) >= 100


@pytest.mark.parametrize("name, summary", auto_docstrings.MAGIC_METHOD_EXTENDED_SUMMARIES.items())
def test_magic_method_summaries_are_multi_sentence(name: str, summary: str) -> None:
    assert _is_multi_sentence(summary), f"Summary for {name!r} must include multiple sentences"


def test_magic_method_fallback_is_multi_sentence() -> None:
    summary = auto_docstrings.extended_summary("function", "__mystery__", "module")
    assert summary == auto_docstrings.DEFAULT_MAGIC_METHOD_FALLBACK
    assert _is_multi_sentence(summary)


@pytest.mark.parametrize(
    "pydantic_helper",
    sorted(PYDANTIC_EXPECTED),
)
def test_pydantic_artifacts_are_documented(pydantic_helper: str) -> None:
    assert pydantic_helper in auto_docstrings.PYDANTIC_ARTIFACT_SUMMARIES


@pytest.mark.parametrize(
    "name, summary",
    auto_docstrings.PYDANTIC_ARTIFACT_SUMMARIES.items(),
)
def test_pydantic_artifact_summaries_are_multi_sentence(name: str, summary: str) -> None:
    assert _is_multi_sentence(summary), f"Summary for {name!r} must include multiple sentences"


def test_pydantic_fallback_is_multi_sentence() -> None:
    summary = auto_docstrings.extended_summary("function", "__pydantic_unknown__", "module")
    assert summary == auto_docstrings.DEFAULT_PYDANTIC_ARTIFACT_SUMMARY
    assert _is_multi_sentence(summary)


def test_generic_function_summary_is_multi_sentence() -> None:
    summary = auto_docstrings.extended_summary("function", "process_records", "module")
    assert _is_multi_sentence(summary)


@pytest.mark.parametrize(
    "source, expected_phrase",
    [
        ("class Example: ...", "data structure"),
        ("class Example(BaseModel): ...", "Pydantic model"),
    ],
)
def test_class_extended_summaries(source: str, expected_phrase: str) -> None:
    node = _parse_class(source)
    summary = auto_docstrings.extended_summary("class", node.name, "pkg.module", node)
    assert expected_phrase in summary
    assert _is_multi_sentence(summary)


def test_module_extended_summary_is_multi_sentence() -> None:
    summary = auto_docstrings.extended_summary("module", "utilities", "pkg.utilities")
    assert _is_multi_sentence(summary)


def test_extended_summary_handles_empty_function_name() -> None:
    summary = auto_docstrings.extended_summary("function", "", "pkg.module")
    assert _is_multi_sentence(summary)


def test_is_pydantic_artifact_detection() -> None:
    assert auto_docstrings.is_pydantic_artifact("__pydantic_unknown__")
    assert auto_docstrings.is_pydantic_artifact("model_dump")
    assert not auto_docstrings.is_pydantic_artifact("not_special")


def test_is_magic_detection() -> None:
    assert auto_docstrings.is_magic("__add__")
    assert not auto_docstrings.is_magic("regular_name")


@pytest.mark.parametrize(
    "kind, name",
    [("module", ""), ("class", ""), ("function", "")],
)
def test_extended_summary_returns_text_for_edge_cases(kind: str, name: str) -> None:
    summary = auto_docstrings.extended_summary(kind, name, "module")
    assert isinstance(summary, str)
    assert summary


def test_extended_summary_uses_default_for_unknown_magic() -> None:
    summary = auto_docstrings.extended_summary("function", "__custom__", "module")
    assert summary == auto_docstrings.DEFAULT_MAGIC_METHOD_FALLBACK


def test_init_summary_is_multi_sentence() -> None:
    summary = auto_docstrings.extended_summary("function", "__init__", "module")
    assert _is_multi_sentence(summary)
