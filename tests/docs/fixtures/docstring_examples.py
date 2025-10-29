"""Sample module used by docstring builder golden tests."""

from __future__ import annotations


def add(value: int, other: int = 0) -> int:
    return value + other


class Example:
    def method(self, value: int) -> int:
        return value
