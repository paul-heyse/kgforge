"""Provide utilities for module.

Auto-generated API documentation for the ``tools.detect_pkg`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.detect_pkg
"""


from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _candidate_names() -> list[str]:
    """Return candidate names.

    Auto-generated reference for the ``_candidate_names`` callable defined in ``tools.detect_pkg``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Raises
    ------
    SystemExit
        Raised when validation fails.
    
    Examples
    --------
    >>> from tools.detect_pkg import _candidate_names
    >>> result = _candidate_names()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.detect_pkg
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return detect packages.

    Auto-generated reference for the ``detect_packages`` callable defined in ``tools.detect_pkg``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.detect_pkg import detect_packages
    >>> result = detect_packages()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.detect_pkg
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    candidates = _candidate_names()
    lowers = [c for c in candidates if c.islower()]
    base = lowers or candidates
    preferred = [c for c in base if "kgfoundry" in c]
    return preferred + [c for c in base if c not in preferred]


def detect_primary() -> str:
    """Return detect primary.

    Auto-generated reference for the ``detect_primary`` callable defined in ``tools.detect_pkg``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from tools.detect_pkg import detect_primary
    >>> result = detect_primary()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.detect_pkg
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    packages = detect_packages()
    return packages[0]


if __name__ == "__main__":
    if "--all" in sys.argv:
        for name in detect_packages():
            print(name)
    else:
        print(detect_primary())
