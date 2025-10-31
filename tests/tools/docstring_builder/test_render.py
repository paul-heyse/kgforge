from __future__ import annotations

from tools.docstring_builder.render import _build_signature, render_docstring
from tools.docstring_builder.schema import DocstringSchema, ParameterDoc


def test_render_docstring_autoescape() -> None:
    schema = DocstringSchema(
        summary="Render <value> & ensure safety",
        parameters=[],
        returns=[],
    )
    rendered = render_docstring(schema, marker="<!-- marker -->", include_signature=False)
    assert "&lt;value&gt;" in rendered
    assert "&amp;" in rendered


def test_build_signature_groups_parameter_kinds() -> None:
    parameters = [
        ParameterDoc(
            name="pos_only",
            annotation="int",
            description="",
            display_name="pos_only",
            kind="positional_only",
        ),
        ParameterDoc(
            name="value",
            annotation="str",
            description="",
            display_name="value",
            kind="positional_or_keyword",
        ),
        ParameterDoc(
            name="args",
            annotation=None,
            description="",
            display_name="*args",
            kind="var_positional",
        ),
        ParameterDoc(
            name="kw",
            annotation="bool",
            description="",
            display_name="kw",
            kind="keyword_only",
            optional=True,
        ),
        ParameterDoc(
            name="kwargs",
            annotation=None,
            description="",
            display_name="**kwargs",
            kind="var_keyword",
        ),
    ]
    schema = DocstringSchema(summary="", parameters=parameters)
    signature = _build_signature(schema)
    assert signature == "(pos_only: int /, value: str, *args, kw: bool = ..., **kwargs)"
