"""Module for kgforge_common.config."""

from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load config.

    Args:
        path (str): TODO.

    Returns:
        Dict[str, Any]: TODO.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
