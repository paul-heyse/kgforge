"""Utilities for asserting dataclass immutability in tests.

These helpers provide a consistent way to verify that frozen dataclasses and
other immutable configuration objects reject attribute mutation. Instead of
relying on ad-hoc casts or direct attribute assignment (which can confuse
static analysers), tests call the helpers defined here to exercise the same
``FrozenInstanceError`` semantics while keeping type-checkers happy.

Example
-------
>>> from tests.helpers.immutability import assert_frozen_attribute
>>> from dataclasses import dataclass
>>> @dataclass(frozen=True)
... class Config:
...     flag: bool = False
>>> config = Config()
>>> assert_frozen_attribute(config, "flag", True)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Protocol

import pytest

__all__ = [
    "assert_frozen_attribute",
    "assert_frozen_attributes",
    "assert_frozen_mutation",
]


def assert_frozen_attribute(obj: object, attr: str, value: object) -> None:
    """Assert that assigning ``value`` to ``attr`` raises ``FrozenInstanceError``."""
    with pytest.raises(FrozenInstanceError):
        setattr(obj, attr, value)


def assert_frozen_attributes(obj: object, **updates: object) -> None:
    """Assert that each attribute in ``updates`` rejects reassignment."""
    for name, value in updates.items():
        assert_frozen_attribute(obj, name, value)


class _Mutator[T](Protocol):
    """Protocol for mutation callables."""

    def __call__(self, obj: T, /) -> object:
        """Mutate object.

        Parameters
        ----------
        obj : T
            Object to mutate.

        Returns
        -------
        object
            Mutation result.
        """
        ...


def assert_frozen_mutation[T](obj: T, mutate: _Mutator[T]) -> None:
    """Assert that executing mutate on obj raises FrozenInstanceError.

    Parameters
    ----------
    obj : T
        Object to mutate.
    mutate : _Mutator[T]
        Callable that mutates the object.
    """
    with pytest.raises(FrozenInstanceError):
        mutate(obj)
