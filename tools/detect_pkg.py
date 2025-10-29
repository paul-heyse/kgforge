"""Overview of detect pkg.

This module bundles detect pkg logic for the kgfoundry stack. It groups
related helpers so downstream packages can import a single cohesive
namespace. Refer to the functions and classes below for implementation
specifics.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _candidate_names() -> list[str]:
    """Compute candidate names.

    Carry out the candidate names operation.

    Returns
    -------
    List[str]
        Description of return value.

    Raises
    ------
    SystemExit
        Raised when validation fails.
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
    """Compute detect packages.

    Carry out the detect packages operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.detect_pkg import detect_packages
    >>> result = detect_packages()
    >>> result  # doctest: +ELLIPSIS
    """
    candidates = _candidate_names()
    lowers = [c for c in candidates if c.islower()]
    base = lowers or candidates
    preferred = [c for c in base if "kgfoundry" in c]
    return preferred + [c for c in base if c not in preferred]


def detect_primary() -> str:
    """Compute detect primary.

    Carry out the detect primary operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.detect_pkg import detect_primary
    >>> result = detect_primary()
    >>> result  # doctest: +ELLIPSIS
    """
    packages = detect_packages()
    return packages[0]


if __name__ == "__main__":
    if "--all" in sys.argv:
        for name in detect_packages():
            print(name)
    else:
        print(detect_primary())
