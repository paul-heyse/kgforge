"""Pytest configuration for search_api tests.

Inherits fixtures and path setup from parent conftest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Set up path for src packages
repo_root = Path(__file__).parent.parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
