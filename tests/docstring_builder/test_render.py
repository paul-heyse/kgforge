"""Tests for docstring rendering utilities."""

from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.schema import DocstringSchema, ParameterDoc


def test_render_signature_includes_none_default() -> None:
    """Parameters defaulting to ``None`` should surface in the signature."""
    parameter = ParameterDoc(
        name="value",
        annotation="int | None",
        description="Describe the value.",
        optional=True,
        default="None",
        display_name="value",
        kind="positional_or_keyword",
    )
    schema = DocstringSchema(summary="Summarize value.", parameters=[parameter])

    docstring = render_docstring(schema, marker="<!-- marker -->", include_signature=True)

    assert "= None" in docstring


"""Tests for docstring rendering behaviour."""


def test_render_docstring_preserves_special_characters() -> None:
    """Docstring rendering should not HTML-escape schema content."""
    schema = DocstringSchema(summary="Return <value> 'untouched'.")
    rendered = render_docstring(schema=schema, marker="[generated]")

    assert "<value>" in rendered
    assert "'untouched'" in rendered
