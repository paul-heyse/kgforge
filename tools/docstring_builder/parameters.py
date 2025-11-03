"""Utilities for handling signature parameter kinds without private inspect access."""

from __future__ import annotations

import inspect
from enum import Enum

__all__ = [
    "ParameterKind",
    "format_parameter_name",
    "normalize_parameter_kind",
]


class ParameterKind(Enum):
    """Public representation of callable parameter kinds."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"

    @property
    def prefix(self) -> str:
        """Return the canonical prefix for rendering the parameter name."""
        if self is ParameterKind.VAR_POSITIONAL:
            return "*"
        if self is ParameterKind.VAR_KEYWORD:
            return "**"
        return ""


_STRING_TO_KIND: dict[str, ParameterKind] = {
    "positional_only": ParameterKind.POSITIONAL_ONLY,
    "positional-only": ParameterKind.POSITIONAL_ONLY,
    "positional_or_keyword": ParameterKind.POSITIONAL_OR_KEYWORD,
    "positional-or-keyword": ParameterKind.POSITIONAL_OR_KEYWORD,
    "var_positional": ParameterKind.VAR_POSITIONAL,
    "variadic_positional": ParameterKind.VAR_POSITIONAL,
    "keyword_only": ParameterKind.KEYWORD_ONLY,
    "var_keyword": ParameterKind.VAR_KEYWORD,
    "variadic_keyword": ParameterKind.VAR_KEYWORD,
}

_INSPECT_TO_KIND: dict[object, ParameterKind] = {
    inspect.Parameter.POSITIONAL_ONLY: ParameterKind.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD: ParameterKind.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.VAR_POSITIONAL: ParameterKind.VAR_POSITIONAL,
    inspect.Parameter.KEYWORD_ONLY: ParameterKind.KEYWORD_ONLY,
    inspect.Parameter.VAR_KEYWORD: ParameterKind.VAR_KEYWORD,
}


def normalize_parameter_kind(value: object) -> ParameterKind:
    """Coerce arbitrary kind objects into :class:`ParameterKind`."""
    if isinstance(value, ParameterKind):
        return value
    if value in _INSPECT_TO_KIND:
        return _INSPECT_TO_KIND[value]
    if isinstance(value, inspect.Parameter):
        return _INSPECT_TO_KIND.get(value.kind, ParameterKind.POSITIONAL_OR_KEYWORD)

    name = _coerce_str(getattr(value, "name", None))
    if name is not None:
        kind = _STRING_TO_KIND.get(name)
        if kind is not None:
            return kind

    token = _coerce_str(getattr(value, "value", None))
    if token is not None:
        kind = _STRING_TO_KIND.get(token)
        if kind is not None:
            return kind

    return ParameterKind.POSITIONAL_OR_KEYWORD


def format_parameter_name(name: str, kind: ParameterKind) -> str:
    """Return the rendered parameter name for docstring sections."""
    return f"{kind.prefix}{name}"


def _coerce_str(value: object | None) -> str | None:
    if isinstance(value, str):
        return value.replace("-", "_").lower()
    return None
