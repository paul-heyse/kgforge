"""Utilities to detect package names for documentation generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _candidate_names() -> list[str]:
    """Collect potential top-level package names."""
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
    """Return package names ordered with a kgfoundry-first preference."""
    candidates = _candidate_names()
    lowers = [c for c in candidates if c.islower()]
    base = lowers or candidates
    preferred = [c for c in base if "kgfoundry" in c]
    return preferred + [c for c in base if c not in preferred]


def detect_primary() -> str:
    """Return the primary package name (first from detect_packages)."""
    packages = detect_packages()
    return packages[0]


if __name__ == "__main__":
    if "--all" in sys.argv:
        for name in detect_packages():
            print(name)
    else:
        print(detect_primary())
