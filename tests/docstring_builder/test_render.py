"""Tests for docstring rendering behaviour."""

from tools.docstring_builder.render import render_docstring
from tools.docstring_builder.schema import DocstringSchema


def test_render_docstring_preserves_special_characters() -> None:
    """Docstring rendering should not HTML-escape schema content."""
    schema = DocstringSchema(summary="Return <value> 'untouched'.")
    rendered = render_docstring(schema=schema, marker="[generated]")

    assert "<value>" in rendered
    assert "'untouched'" in rendered
