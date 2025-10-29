"""Render docstrings from schemas using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, StrictUndefined
from tools.docstring_builder.schema import DocstringSchema

_TEMPLATE = """{{ schema.summary }}\n{{ marker }}\n{% if schema.extended %}\n{{ schema.extended }}\n{% endif %}{% if schema.parameters %}\nParameters\n----------\n{% for parameter in schema.parameters %}{{ parameter.name }} : {{ parameter.annotation or 'Any' }}{% if parameter.optional %}, optional{% endif %}{% if parameter.default %}, by default {{ parameter.default }}{% endif %}\n    {{ parameter.description }}\n{% endfor %}{% endif %}{% if schema.returns %}\n{% set has_yields = schema.returns|selectattr('kind', 'equalto', 'yields')|list|length > 0 %}{% if has_yields %}Yields\n------\n{% else %}Returns\n-------\n{% endif %}{% for entry in schema.returns %}{{ entry.annotation or 'Any' }}\n    {{ entry.description }}\n{% endfor %}{% endif %}{% if schema.raises %}\nRaises\n------\n{% for entry in schema.raises %}{{ entry.exception }}\n    {{ entry.description }}\n{% endfor %}{% endif %}{% if schema.notes %}\nNotes\n-----\n{% for note in schema.notes %}{{ note }}\n{% endfor %}{% endif %}{% if schema.see_also %}\nSee Also\n--------\n{% for link in schema.see_also %}{{ link }}\n{% endfor %}{% endif %}{% if schema.examples %}\nExamples\n--------\n{% for example in schema.examples %}{{ example }}\n{% endfor %}{% endif %}"""

_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
_TEMPLATE_OBJ = _ENV.from_string(_TEMPLATE)


def render_docstring(schema: DocstringSchema, marker: str) -> str:
    """Render the provided schema to a concrete docstring string."""
    rendered = _TEMPLATE_OBJ.render(schema=schema, marker=marker).strip()
    return rendered + "\n"


def write_template_to(path: Path) -> None:
    """Persist the default template to disk for debugging purposes."""
    path.write_text(_TEMPLATE, encoding="utf-8")


__all__ = ["render_docstring", "write_template_to"]
