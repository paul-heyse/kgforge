"""Render docstrings from schemas using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, StrictUndefined
from tools.docstring_builder.schema import DocstringSchema

_TEMPLATE = """{{ schema.summary }}\n\n{{ marker }}{% if signature %}\n\nSignature\n---------\n{{ signature }}{% endif %}{% if schema.extended %}\n\n{{ schema.extended }}{% endif %}{% if schema.parameters %}\n\nParameters\n----------\n{% for parameter in schema.parameters %}{{ parameter.display_name or parameter.name }} : {{ parameter.annotation or 'Any' }}{% if parameter.optional %}, optional{% endif %}{% if parameter.default %}, by default {{ parameter.default }}{% endif %}\n    {{ parameter.description or 'Description forthcoming.' }}\n{% endfor %}{% endif %}{% if schema.returns %}\n\n{% set has_yields = schema.returns|selectattr('kind', 'equalto', 'yields')|list|length > 0 %}{% if has_yields %}Yields\n------\n{% else %}Returns\n-------\n{% endif %}{% for entry in schema.returns %}{{ entry.annotation or 'Any' }}\n    {{ entry.description or 'Description forthcoming.' }}\n{% endfor %}{% endif %}{% if schema.raises %}\n\nRaises\n------\n{% for entry in schema.raises %}{{ entry.exception }}\n    {{ entry.description or 'Description forthcoming.' }}\n{% endfor %}{% endif %}{% if schema.notes %}\n\nNotes\n-----\n{% for note in schema.notes %}{{ note }}\n{% endfor %}{% endif %}{% if schema.see_also %}\n\nSee Also\n--------\n{% for link in schema.see_also %}{{ link }}\n{% endfor %}{% endif %}{% if schema.examples %}\n\nExamples\n--------\n{% for example in schema.examples %}{{ example }}\n{% endfor %}{% endif %}"""

_ENV = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=True)
_TEMPLATE_OBJ = _ENV.from_string(_TEMPLATE)


def render_docstring(schema: DocstringSchema, marker: str, include_signature: bool = False) -> str:
    """Render the provided schema to a concrete docstring string."""
    signature = _build_signature(schema) if include_signature else ""
    rendered = _TEMPLATE_OBJ.render(schema=schema, marker=marker, signature=signature).strip()
    return rendered + "\n"


def write_template_to(path: Path) -> None:
    """Persist the default template to disk for debugging purposes."""
    path.write_text(_TEMPLATE, encoding="utf-8")


def _build_signature(schema: DocstringSchema) -> str:
    parameters = schema.parameters
    if not parameters:
        return ""
    parts: list[str] = []
    pos_only: list[str] = []
    pos_or_kw: list[str] = []
    kw_only: list[str] = []
    var_positional: str | None = None
    var_keyword: str | None = None
    for parameter in parameters:
        token = parameter.display_name or parameter.name
        annotation = parameter.annotation
        entry = token
        if annotation:
            entry += f": {annotation}"
        default = parameter.default
        if default is not None:
            entry += f" = {default}"
        elif parameter.optional:
            entry += " = ..."
        kind = parameter.kind
        if kind == "positional_only":
            pos_only.append(entry)
        elif kind == "positional_or_keyword":
            pos_or_kw.append(entry)
        elif kind == "var_positional":
            var_positional = entry
        elif kind == "keyword_only":
            kw_only.append(entry)
        elif kind == "var_keyword":
            var_keyword = entry
        else:
            pos_or_kw.append(entry)
    if pos_only:
        parts.append(", ".join(pos_only) + " /")
    if pos_or_kw:
        parts.append(", ".join(pos_or_kw))
    if var_positional:
        parts.append(var_positional)
    if kw_only:
        if not var_positional:
            parts.append("*")
        parts.append(", ".join(kw_only))
    if var_keyword:
        parts.append(var_keyword)
    signature = ", ".join(part for part in parts if part)
    return f"({signature})"


__all__ = ["render_docstring", "write_template_to"]
