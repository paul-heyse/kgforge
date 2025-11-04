"""Shared test helpers for kgfoundry.

This package centralises fixtures and assertion utilities used across the
test-suite. Helpers defined here should have zero runtime side-effects and
favour explicit, typed APIs so that they compose cleanly with the standards
documented in ``AGENTS.md``.
"""

from __future__ import annotations

from tests.helpers.immutability import (
    assert_frozen_attribute,
    assert_frozen_attributes,
    assert_frozen_mutation,
)

__all__ = [
    "assert_frozen_attribute",
    "assert_frozen_attributes",
    "assert_frozen_mutation",
]
