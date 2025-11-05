"""Utility helpers shared across docstring builder modules."""

from __future__ import annotations


def optional_str(value: str | None) -> str | None:
    """Return ``value`` when truthy, otherwise ``None``.

    Parameters
    ----------
    value : str | None
        Candidate string value.

    Returns
    -------
    str | None
        ``value`` when truthy, otherwise ``None``.
    """
    return value if value else None


def optional_str_list(values: list[str] | tuple[str, ...] | None) -> list[str] | None:
    """Return a list of ``values`` or ``None`` when empty.

    Parameters
    ----------
    values : list[str] | tuple[str, ...] | None
        Candidate collection of strings.

    Returns
    -------
    list[str] | None
        Materialised list when non-empty, otherwise ``None``.
    """
    if values is None:
        return None
    materialised = list(values)
    return materialised if materialised else None
