"""Render docstring schemas into normalized textual docstrings."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from jinja2 import Environment, StrictUndefined

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from jinja2 import Template, Undefined
    from jinja2.utils import select_autoescape

    from tools.docstring_builder.schema import DocstringSchema, ParameterDoc
else:
    Undefined = StrictUndefined
    try:
        from jinja2.utils import select_autoescape
    except ImportError:
        import typing

        # Fallback for older jinja2 versions
        def select_autoescape(**_kwargs: object) -> typing.Callable[[str | None], bool]:
            return lambda _filename=None: False


_TEMPLATE = """{{ schema.summary }}\n\n{{ marker }}{% if signature %}\n\nSignature\n---------\n{{ signature }}{% endif %}{% if schema.extended %}\n\n{{ schema.extended }}{% endif %}{% if schema.parameters %}\n\nParameters\n----------\n{% for parameter in schema.parameters %}{{ parameter.display_name or parameter.name }} : {{ parameter.annotation or 'Any' }}{% if parameter.optional %}, optional{% endif %}{% if parameter.default is not none %}, by default {{ parameter.default }}{% endif %}\n    {{ parameter.description or 'Description forthcoming.' }}\n{% endfor %}{% endif %}{% if schema.returns %}\n\n{% set has_yields = schema.returns|selectattr('kind', 'equalto', 'yields')|list|length > 0 %}{% if has_yields %}Yields\n------\n{% else %}Returns\n-------\n{% endif %}{% for entry in schema.returns %}{{ entry.annotation or 'Any' }}\n    {{ entry.description or 'Description forthcoming.' }}\n{% endfor %}{% endif %}{% if schema.raises %}\n\nRaises\n------\n{% for entry in schema.raises %}{{ entry.exception }}\n    {{ entry.description or 'Description forthcoming.' }}\n{% endfor %}{% endif %}{% if schema.notes %}\n\nNotes\n-----\n{% for note in schema.notes %}{{ note }}\n{% endfor %}{% endif %}{% if schema.see_also %}\n\nSee Also\n--------\n{% for link in schema.see_also %}{{ link }}\n{% endfor %}{% endif %}{% if schema.examples %}\n\nExamples\n--------\n{% for example in schema.examples %}{{ example }}\n{% endfor %}{% endif %}"""


def _build_environment() -> Environment:
    undefined_cls: type[Undefined] = StrictUndefined
    return Environment(
        undefined=undefined_cls,
        trim_blocks=False,
        lstrip_blocks=True,
        autoescape=select_autoescape(
            enabled_extensions=(), default=False, default_for_string=False
        ),
    )


_ENV = _build_environment()
_TEMPLATE_OBJ: Template = _ENV.from_string(_TEMPLATE)


def render_docstring(
    schema: DocstringSchema,
    marker: str,
    *,
    include_signature: bool = False,
) -> str:
    """Render the provided schema to a concrete docstring string."""
    signature = _build_signature(schema) if include_signature else ""
    rendered = _TEMPLATE_OBJ.render(schema=schema, marker=marker, signature=signature)
    return rendered.strip() + "\n"


def write_template_to(path: Path) -> None:
    """Persist the default template to disk for debugging purposes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE, encoding="utf-8")


def _build_signature(schema: DocstringSchema) -> str:
    parameters = schema.parameters
    if not parameters:
        return ""
    formatted = list(_format_parameters(parameters))
    grouped = _group_parameters(formatted)
    return _compose_signature(grouped)


def _format_parameters(parameters: Iterable[ParameterDoc]) -> Iterable[tuple[str, str]]:
    """Yield parameter kind and formatted signature entry for ``parameters``."""
    for parameter in parameters:
        token = parameter.display_name or parameter.name
        entry = token
        if parameter.annotation:
            entry += f": {parameter.annotation}"
        if parameter.default is not None:
            entry += f" = {parameter.default}"
        elif parameter.optional:
            entry += " = ..."
        yield parameter.kind, entry


def _group_parameters(
    formatted: Iterable[tuple[str, str]],
) -> dict[str, list[str] | str | None]:
    """Group formatted parameters by signature position semantics."""
    groups: dict[str, list[str] | str | None] = {
        "positional_only": [],
        "positional_or_keyword": [],
        "keyword_only": [],
        "var_positional": None,
        "var_keyword": None,
    }

    def _append(kind: str, value: str) -> None:
        bucket = groups[kind]
        if isinstance(bucket, list):
            bucket.append(value)

    list_kinds = {"positional_only", "positional_or_keyword", "keyword_only"}
    for kind, entry in formatted:
        if kind in list_kinds:
            _append(kind, entry)
            continue
        if kind == "var_positional":
            groups["var_positional"] = entry
            continue
        if kind == "var_keyword":
            groups["var_keyword"] = entry
            continue
        _append("positional_or_keyword", entry)
    return groups  # complexity acceptable for parameter grouping logic


def _compose_signature(groups: dict[str, list[str] | str | None]) -> str:
    """Return the rendered signature string for grouped parameters."""
    parts: list[str] = []
    pos_only = cast("list[str]", groups.get("positional_only", []))
    if pos_only:
        parts.append(", ".join(pos_only) + " /")

    pos_or_kw = cast("list[str]", groups.get("positional_or_keyword", []))
    if pos_or_kw:
        parts.append(", ".join(pos_or_kw))

    var_positional = cast("str | None", groups.get("var_positional"))
    if var_positional:
        parts.append(var_positional)

    kw_only = cast("list[str]", groups.get("keyword_only", []))
    if kw_only:
        if not var_positional:
            parts.append("*")
        parts.append(", ".join(kw_only))

    var_keyword = cast("str | None", groups.get("var_keyword"))
    if var_keyword:
        parts.append(var_keyword)

    signature = ", ".join(part for part in parts if part)
    return f"({signature})"


__all__ = ["render_docstring", "write_template_to"]
