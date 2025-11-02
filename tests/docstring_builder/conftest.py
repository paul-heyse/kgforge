"""Shared test configuration for docstring builder tests."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[2] / "src"
_SRC_STR = str(_SRC_PATH)
if _SRC_STR not in sys.path:
    sys.path.insert(0, _SRC_STR)
