#!/usr/bin/env python
"""Provide utilities for module.

Auto-generated API documentation for the ``tools.generate_docstrings`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.generate_docstrings
"""


from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC_TEMPLATES = REPO / "tools" / "doq_templates" / "numpy"
TARGETS = [
    REPO / "src",
    REPO / "tools",
    REPO / "docs" / "_scripts",
]
LOG_DIR = REPO / "site" / "_build" / "docstrings"
LOG_FILE = LOG_DIR / "fallback.log"


def has_python_files(path: Path) -> bool:
    """Return has python files.

    Auto-generated reference for the ``has_python_files`` callable defined in ``tools.generate_docstrings``.
    
    Parameters
    ----------
    path : Path
        Description for ``path``.
    
    Returns
    -------
    bool
        Description of return value.
    
    Examples
    --------
    >>> from tools.generate_docstrings import has_python_files
    >>> result = has_python_files(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.generate_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    return any(path.rglob("*.py"))


def run_doq(target: Path) -> None:
    """Return run doq.

    Auto-generated reference for the ``run_doq`` callable defined in ``tools.generate_docstrings``.
    
    Parameters
    ----------
    target : Path
        Description for ``target``.
    
    Examples
    --------
    >>> from tools.generate_docstrings import run_doq
    >>> run_doq(...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.generate_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    cmd = [
        sys.executable,
        "-m",
        "doq.cli",
        "--formatter",
        "numpy",
        "-t",
        str(DOC_TEMPLATES),
        "-w",
        "-r",
        "-d",
        str(target),
    ]
    subprocess.run(cmd, check=True)


def run_fallback(target: Path) -> None:
    """Return run fallback.

    Auto-generated reference for the ``run_fallback`` callable defined in ``tools.generate_docstrings``.
    
    Parameters
    ----------
    target : Path
        Description for ``target``.
    
    Examples
    --------
    >>> from tools.generate_docstrings import run_fallback
    >>> run_fallback(...)  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.generate_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    cmd = [
        sys.executable,
        "tools/auto_docstrings.py",
        "--target",
        str(target),
        "--log",
        str(LOG_FILE),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    """Return main.

    Auto-generated reference for the ``main`` callable defined in ``tools.generate_docstrings``.
    
    Examples
    --------
    >>> from tools.generate_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.generate_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        if not target.exists() or not target.is_dir():
            continue
        if not has_python_files(target):
            continue
        print(f"[docstrings] Updating {target.relative_to(REPO)}")
        run_doq(target)
        run_fallback(target)


if __name__ == "__main__":
    main()
