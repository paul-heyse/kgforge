"""Testing bootstrap helpers for kgfoundry.

This module centralises environment setup so pytest (and any other test
runner) can resolve the project package layout without ad-hoc hacks. It is
imported by ``tests.conftest`` before any of the test modules, ensuring the
``src`` directory is always on ``sys.path`` and optional dependencies can be
loaded lazily.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final, cast

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SRC_PATH: Final[Path] = REPO_ROOT / "src"


class _BootstrapState:
    bootstrapped: bool = False


def ensure_src_path() -> None:
    """Add the ``src`` directory to ``sys.path`` exactly once."""
    if _BootstrapState.bootstrapped:
        return
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
        importlib.invalidate_caches()
    _BootstrapState.bootstrapped = True


def load_optional_module(module_name: str) -> ModuleType | None:
    """Return ``module_name`` when importable, otherwise ``None``."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


def load_optional_attr(module_name: str, attr_name: str) -> object | None:
    """Return ``getattr`` from ``module_name`` when available.

    ``None`` is returned when either the module or the attribute cannot be
    imported. Consumers remain responsible for type-casting the result.
    """
    module = load_optional_module(module_name)
    if module is None:
        return None
    return cast(object | None, getattr(module, attr_name, None))


# Ensure the src layout is active as soon as the bootstrap module is imported.
ensure_src_path()
