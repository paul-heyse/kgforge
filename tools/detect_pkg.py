"""Identify canonical kgfoundry packages for repository automation.

This script scans the repository layout to discover eligible package directories and orders them so
helper tools can consistently select a primary target. It mirrors the behaviour of the historical
shell script while filtering out non-package utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _candidate_names() -> list[str]:
    """Collect package names discoverable from the repository layout.

    The search considers ``src/<package>/__init__.py`` as well as top-level
    packages that expose an ``__init__`` file. Utility directories such as
    ``docs`` and ``tools`` are ignored.

    Returns
    -------
    list[str]
        Sorted list of unique package directory names.

    Raises
    ------
    SystemExit
        Raised when no package candidates are found, mirroring the behaviour of
        the historical shell script this tool replaces.
    """
    names: set[str] = set()
    if SRC.exists():
        names.update(p.parent.name for p in SRC.glob("*/__init__.py"))
    names.update(
        p.parent.name
        for p in ROOT.glob("*/__init__.py")
        if p.parent.name not in {"docs", "tools", "optional"}
    )
    if not names:
        raise SystemExit(1)
    names.discard("src")
    return sorted(names)


def detect_packages() -> list[str]:
    """Return preferred package names ordered by namespace relevance.

    Packages containing ``kgfoundry`` bubble to the front, followed by other
    lowercase candidates. The ordering matches the expectations of scripts that
    need a deterministic primary package name.

    Returns
    -------
    list[str]
        Package names ordered by preference.
    """
    candidates = _candidate_names()
    lowers = [c for c in candidates if c.islower()]
    base = lowers or candidates
    preferred = [c for c in base if "kgfoundry" in c]
    return preferred + [c for c in base if c not in preferred]


def detect_primary() -> str:
    """Return the first preferred package name.

    Returns
    -------
    str
        The canonical package name used by tooling that only supports a single
        package.
    """
    packages = detect_packages()
    return packages[0]


if __name__ == "__main__":
    if "--all" in sys.argv:
        for name in detect_packages():
            sys.stdout.write(f"{name}\n")
    else:
        sys.stdout.write(f"{detect_primary()}\n")
