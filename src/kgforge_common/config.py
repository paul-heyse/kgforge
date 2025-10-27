"""Module for kgforge_common.config."""

from __future__ import annotations
from typing import Any, Dict
import os, yaml

def load_config(path: str) -> Dict[str, Any]:
    """Load config.

    Args:
        path (str): TODO.

    Returns:
        Dict[str, Any]: TODO.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
