"""Test imports."""

from __future__ import annotations

from kgfoundry_common.config import AppSettings


def test_import():
    s = AppSettings()
    assert s.log_level == "INFO"
