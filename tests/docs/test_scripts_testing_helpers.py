from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from docs.scripts import (
    build_symbol_index,
    mkdocs_gen_api,
    shared,
    symbol_delta,
    validate_artifacts,
)
from docs.scripts.testing import clear_lazy_import_caches

if TYPE_CHECKING:
    import pytest


def test_clear_lazy_import_caches_invokes_all_facade_clearers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that clear_lazy_import_caches invokes all facade clear_cache methods.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    """
    facades = (
        shared,
        build_symbol_index,
        mkdocs_gen_api,
        symbol_delta,
        validate_artifacts,
    )
    mocks: list[MagicMock] = []
    for module in facades:
        mock = MagicMock()
        monkeypatch.setattr(module, "clear_cache", mock)
        mocks.append(mock)

    clear_lazy_import_caches()

    for mock in mocks:
        mock.assert_called_once_with()
