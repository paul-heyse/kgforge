"""Typed introspection module for plugin factory validation.

This module provides a typed wrapper around Python's inspect module
to eliminate the 'Any' types that come from its type stubs.
"""

from __future__ import annotations

import inspect as stdlib_inspect
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "ParameterInfo",
    "get_signature",
    "has_required_parameters",
]


@dataclass(frozen=True, slots=True)
class ParameterInfo:
    """Type-safe parameter information without Any types."""

    name: str
    """Name of the parameter."""

    has_default: bool
    """True if parameter has a default value."""

    is_var_positional: bool
    """True if parameter accepts *args."""

    is_var_keyword: bool
    """True if parameter accepts **kwargs."""


def get_signature(func: Callable[..., object]) -> list[ParameterInfo]:
    """Get typed parameter information without Any types.

    Parameters
    ----------
    func : Callable
        The function or class to inspect.

    Returns
    -------
    list[ParameterInfo]
        List of parameter information objects.

    Raises
    ------
    ValueError
        If signature cannot be obtained.
    """
    try:
        sig = stdlib_inspect.signature(func)
    except (ValueError, TypeError) as exc:
        message = f"Could not inspect signature for {func}"
        raise ValueError(message) from exc

    params: list[ParameterInfo] = []
    for param_name, param in sig.parameters.items():
        # Get attributes without accessing Any-typed values directly
        has_default = param.default is not stdlib_inspect.Parameter.empty
        is_var_pos = param.kind is stdlib_inspect.Parameter.VAR_POSITIONAL
        is_var_kw = param.kind is stdlib_inspect.Parameter.VAR_KEYWORD

        params.append(
            ParameterInfo(
                name=param_name,
                has_default=bool(has_default),
                is_var_positional=bool(is_var_pos),
                is_var_keyword=bool(is_var_kw),
            )
        )

    return params


def has_required_parameters(func: Callable[..., object]) -> bool:
    """Check if a callable has required (non-default, non-*args/**kwargs) parameters.

    Parameters
    ----------
    func : Callable
        The function or class to check.

    Returns
    -------
    bool
        True if the callable has required parameters.

    Raises
    ------
    ValueError
        If signature cannot be obtained.
    """
    params = get_signature(func)

    for param in params:
        # Skip 'self' parameter for methods
        if param.name == "self":
            continue

        # Check if this parameter is required
        is_required = (
            not param.has_default and not param.is_var_positional and not param.is_var_keyword
        )

        if is_required:
            return True

    return False
