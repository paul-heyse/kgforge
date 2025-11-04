"""Pytest configuration for docs-specific tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from docs.scripts.testing import clear_lazy_import_caches

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_docs_scripts_caches() -> Iterator[None]:
    """Ensure docs script caches are clean around each test."""
    clear_lazy_import_caches()
    try:
        yield
    finally:
        clear_lazy_import_caches()
